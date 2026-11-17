#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

use core::fmt::Write;

use nexashift::math::{HmdMatrix34_t, SimdMath, Vec3};
use nexashift::openvr::{OpenVrError, OpenVrSession};
use nexashift::platform::{
    MonotonicClock, ReactiveWaiter, StackBuf, command_line_has_flag, install_signal_handlers,
    is_running, query_chaperone_file_stamp, query_process_stats, write_stderr, write_stdout,
};
use nexashift::state::{DragPhase, DragStateMachine, HandInput, InputFrame, StepAction};

#[cfg(all(not(test), target_os = "windows"))]
#[link(name = "vcruntime")]
#[link(name = "msvcrt")]
unsafe extern "C" {}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    nexashift::platform::abort()
}

fn print_out(args: core::fmt::Arguments<'_>) {
    let mut buf = StackBuf::<512>::new();
    let _ = buf.write_fmt(args);
    write_stdout(buf.as_bytes());
}

fn print_err(args: core::fmt::Arguments<'_>) {
    let mut buf = StackBuf::<512>::new();
    let _ = buf.write_fmt(args);
    write_stderr(buf.as_bytes());
}

fn print_openvr_error(err: OpenVrError) {
    match err {
        OpenVrError::LibraryNotFound => {
            print_err(format_args!("error: could not locate openvr_api library\n"));
        }
        OpenVrError::MissingEntryPoint(sym) => {
            print_err(format_args!("error: missing OpenVR export {}\n", sym));
        }
        OpenVrError::VrInitFailed(code) => {
            print_err(format_args!("error: VR_InitInternal failed (EVRInitError={})\n", code));
        }
        OpenVrError::InterfaceFailed(name, code) => {
            print_err(format_args!(
                "error: VR_GetGenericInterface({}) failed (EVRInitError={})\n",
                name, code
            ));
        }
        OpenVrError::ManifestNotFound => {
            print_err(format_args!("error: could not locate manifest/actions.json\n"));
        }
        OpenVrError::InputInitFailed(stage, code) => {
            print_err(format_args!(
                "error: IVRInput::{} failed (EVRInputError={})\n",
                stage, code
            ));
        }
    }
}

fn mat_translation_close(a: &HmdMatrix34_t, b: &HmdMatrix34_t, eps: f32) -> bool {
    let mut r = 0;
    while r < 3 {
        let mut c = 0;
        while c < 4 {
            if (a.m[r][c] - b.m[r][c]).abs() > eps {
                return false;
            }
            c += 1;
        }
        r += 1;
    }
    true
}

#[inline]
fn wait_next_vr_frame(session: &OpenVrSession, waiter: &ReactiveWaiter) {
    if !session.wait_frame_sync(100) && is_running() {
        waiter.wait_stop_or_timeout_ms(100);
    }
}

fn run_poc(simd: SimdMath) -> i32 {
    install_signal_handlers();

    let session = match OpenVrSession::init(true) {
        Ok(s) => s,
        Err(OpenVrError::ManifestNotFound | OpenVrError::InputInitFailed(_, _)) => {
            match OpenVrSession::init(false) {
                Ok(s) => s,
                Err(e) => {
                    print_openvr_error(e);
                    return 1;
                }
            }
        }
        Err(e) => {
            print_openvr_error(e);
            return 1;
        }
    };

    let Some(m0) = session.get_working_standing_pose() else {
        print_err(format_args!(
            "error: failed to read initial working standing zero pose\n"
        ));
        return 1;
    };

    let t0 = m0.translation();
    print_out(format_args!(
        "poc: connected (HMD target {} Hz, simd={}), initial translation=({:.4}, {:.4}, {:.4})\n",
        session.hmd_refresh_hz() as u32,
        simd.level_name(),
        t0.x,
        t0.y,
        t0.z
    ));

    let clock = MonotonicClock::new();
    let waiter = ReactiveWaiter::new();
    let stats_before = query_process_stats();
    let chap_before = query_chaperone_file_stamp();

    let steps: u64 = 120;
    let mut min_ns = u64::MAX;
    let mut max_ns = 0u64;
    let mut sum_ns = 0u64;
    let mut verified_steps = 0u64;

    let mut i = 1u64;
    while i <= steps && is_running() {
        let frac = (i as f32) / (steps as f32);
        // Smooth out-and-back excursion in all 3 axes (0 -> 0.05m -> 0m).
        let wave = if frac <= 0.5 {
            frac * 2.0
        } else {
            (1.0 - frac) * 2.0
        };
        let delta = Vec3::new(0.05 * wave, 0.03 * wave, -0.05 * wave);
        let target = simd.apply_raw_translation(&m0, delta);

        let t_start = clock.now_ns();
        session.apply_working_standing_pose(&target);
        let dt = clock.now_ns().saturating_sub(t_start);

        if dt < min_ns {
            min_ns = dt;
        }
        if dt > max_ns {
            max_ns = dt;
        }
        sum_ns = sum_ns.saturating_add(dt);

        if let Some(readback) = session.peek_working_standing_pose() {
            if mat_translation_close(&readback, &target, 1e-5) {
                verified_steps += 1;
            }
        }

        wait_next_vr_frame(&session, &waiter);
        i += 1;
    }

    // Restore original M0 cleanly.
    session.apply_working_standing_pose(&m0);
    let restored_ok = match session.peek_working_standing_pose() {
        Some(rb) => mat_translation_close(&rb, &m0, 1e-5),
        None => false,
    };

    let chap_after = query_chaperone_file_stamp();
    let chaperone_unmodified = chap_before.is_some() && chap_before == chap_after;
    let _ = session.commit_working_standing_pose(&m0);
    let stats_after = query_process_stats();
    let ipc_write_bytes = stats_after.write_bytes.saturating_sub(stats_before.write_bytes);
    let avg_us = (sum_ns / steps) as f64 / 1000.0;
    let min_us = min_ns as f64 / 1000.0;
    let max_us = max_ns as f64 / 1000.0;

    print_out(format_args!(
        "poc: verified {}/{} live preview steps, restored={}, chaperone_unmodified={}, ipc_write_bytes={}\n",
        verified_steps, steps, restored_ok, chaperone_unmodified, ipc_write_bytes
    ));
    print_out(format_args!(
        "poc: apply+preview latency us: min={:.1}, avg={:.1}, max={:.1}\n",
        min_us, avg_us, max_us
    ));

    if verified_steps == steps && restored_ok && chaperone_unmodified {
        0
    } else {
        1
    }
}

fn run_bench(simd: SimdMath) -> i32 {
    install_signal_handlers();

    let session = match OpenVrSession::init(true) {
        Ok(s) => s,
        Err(e) => {
            print_openvr_error(e);
            return 1;
        }
    };

    print_out(format_args!(
        "bench[init]: simd={}, hmd_hz={}\n",
        simd.level_name(),
        session.hmd_refresh_hz() as u32
    ));

    let clock = MonotonicClock::new();
    let waiter = ReactiveWaiter::new();
    let mut sm = DragStateMachine::with_simd(simd);

    // 1. Idle benchmark (5 seconds at low-power ~30 Hz reactive wakeups).
    let bench_duration_ns: u64 = 5_000_000_000;
    let idle_chap_before = query_chaperone_file_stamp();
    let idle_start_stats = query_process_stats();
    let idle_t0 = clock.now_ns();
    let mut idle_wakeups: u64 = 0;

    while is_running() && clock.now_ns().saturating_sub(idle_t0) < bench_duration_ns {
        waiter.wait_stop_or_timeout_ms(33);
        if !is_running() || ((idle_wakeups & 15) == 0 && session.poll_should_quit()) {
            break;
        }
        let frame = session.sample_inputs(sm.active_device_hint());
        let _ = sm.step_with(frame, || session.get_working_standing_pose());
        idle_wakeups += 1;
    }

    let idle_wall_ns = clock.now_ns().saturating_sub(idle_t0).max(1);
    let idle_end_stats = query_process_stats();
    let idle_chap_after = query_chaperone_file_stamp();
    let idle_chap_modified = idle_chap_before != idle_chap_after;
    let idle_cpu_ns = idle_end_stats
        .cpu_time_ns
        .saturating_sub(idle_start_stats.cpu_time_ns);
    let idle_cpu_pct_1core = (idle_cpu_ns as f64) * 100.0 / (idle_wall_ns as f64);
    let idle_cpu_pct_12t = idle_cpu_pct_1core / 12.0;
    let idle_wakeups_sec = (idle_wakeups as f64) * 1_000_000_000.0 / (idle_wall_ns as f64);
    let idle_ipc_writes = idle_end_stats
        .write_bytes
        .saturating_sub(idle_start_stats.write_bytes);

    print_out(format_args!(
        "bench[idle]: wall_ms={:.1}, wakeups={} ({:.1}/s), cpu_ms={:.3}, cpu_1core={:.4}%, cpu_12t={:.4}%, private_kb={}, ipc_write_bytes={}, chaperone_modified={}\n",
        idle_wall_ns as f64 / 1_000_000.0,
        idle_wakeups,
        idle_wakeups_sec,
        idle_cpu_ns as f64 / 1_000_000.0,
        idle_cpu_pct_1core,
        idle_cpu_pct_12t,
        idle_end_stats.private_bytes / 1024,
        idle_ipc_writes,
        idle_chap_modified
    ));

    // 2. Reactive Dragging benchmark (5 seconds synchronized by WaitFrameSync).
    let Some(m0) = session.get_working_standing_pose() else {
        print_err(format_args!("error: failed to read working standing pose\n"));
        return 1;
    };

    let mut drag_sm = DragStateMachine::with_simd(simd);
    let drag_chap_before = query_chaperone_file_stamp();
    let drag_start_stats = query_process_stats();
    let drag_t0 = clock.now_ns();
    let mut drag_wakeups: u64 = 0;
    let mut lat_min_ns = u64::MAX;
    let mut lat_max_ns = 0u64;
    let mut lat_sum_ns = 0u64;

    while is_running() && clock.now_ns().saturating_sub(drag_t0) < bench_duration_ns {
        wait_next_vr_frame(&session, &waiter);
        if !is_running() || ((drag_wakeups & 31) == 0 && session.poll_should_quit()) {
            break;
        }
        let iter_t0 = clock.now_ns();
        let _real_frame = session.sample_inputs(drag_sm.active_device_hint());
        let phase = ((drag_wakeups % 120) as f32) / 120.0;
        let offset = if phase <= 0.5 {
            phase * 0.04
        } else {
            (1.0 - phase) * 0.04
        };
        let sim_frame = InputFrame {
            left: HandInput {
                pressed: true,
                device_index: 1,
                pose_valid: true,
                position_raw: Vec3::new(offset, 1.0 + offset, -offset),
            },
            right: HandInput::UNPRESSED,
        };
        if let StepAction::UpdatePlayspace(m) = drag_sm.step_with(sim_frame, || Some(m0)) {
            session.apply_working_standing_pose(&m);
        }
        let iter_dt = clock.now_ns().saturating_sub(iter_t0);
        if iter_dt < lat_min_ns {
            lat_min_ns = iter_dt;
        }
        if iter_dt > lat_max_ns {
            lat_max_ns = iter_dt;
        }
        lat_sum_ns = lat_sum_ns.saturating_add(iter_dt);

        drag_wakeups += 1;
    }

    // Restore original M0 cleanly.
    session.apply_working_standing_pose(&m0);

    let drag_wall_ns = clock.now_ns().saturating_sub(drag_t0).max(1);
    let drag_end_stats = query_process_stats();
    let drag_chap_after = query_chaperone_file_stamp();
    let drag_chap_modified = drag_chap_before != drag_chap_after;
    let _ = session.commit_working_standing_pose(&m0);
    let drag_cpu_ns = drag_end_stats
        .cpu_time_ns
        .saturating_sub(drag_start_stats.cpu_time_ns);
    let drag_cpu_pct_1core = (drag_cpu_ns as f64) * 100.0 / (drag_wall_ns as f64);
    let drag_cpu_pct_12t = drag_cpu_pct_1core / 12.0;
    let drag_wakeups_sec = (drag_wakeups as f64) * 1_000_000_000.0 / (drag_wall_ns as f64);
    let drag_ipc_writes = drag_end_stats
        .write_bytes
        .saturating_sub(drag_start_stats.write_bytes);
    let lat_avg_us = (lat_sum_ns / drag_wakeups.max(1)) as f64 / 1000.0;

    print_out(format_args!(
        "bench[drag]: wall_ms={:.1}, wakeups={} ({:.1}/s), cpu_ms={:.3}, cpu_1core={:.4}%, cpu_12t={:.4}%, private_kb={}, ipc_write_bytes={}, chaperone_modified={}\n",
        drag_wall_ns as f64 / 1_000_000.0,
        drag_wakeups,
        drag_wakeups_sec,
        drag_cpu_ns as f64 / 1_000_000.0,
        drag_cpu_pct_1core,
        drag_cpu_pct_12t,
        drag_end_stats.private_bytes / 1024,
        drag_ipc_writes,
        drag_chap_modified
    ));
    print_out(format_args!(
        "bench[drag]: sample+step+preview latency us: min={:.1}, avg={:.1}, max={:.1}\n",
        lat_min_ns as f64 / 1000.0,
        lat_avg_us,
        lat_max_ns as f64 / 1000.0
    ));

    0
}

fn run_daemon(simd: SimdMath) -> i32 {
    install_signal_handlers();

    let session = match OpenVrSession::init(true) {
        Ok(s) => s,
        Err(e) => {
            print_openvr_error(e);
            return 1;
        }
    };

    let clock = MonotonicClock::new();
    let waiter = ReactiveWaiter::new();
    let mut sm = DragStateMachine::with_simd(simd);
    let mut tick: u64 = 0;
    let mut last_drag_release_ns: Option<u64> = None;

    while is_running() {
        if sm.is_dragging() {
            wait_next_vr_frame(&session, &waiter);
        } else {
            waiter.wait_stop_or_timeout_ms(33);
        }
        if !is_running() {
            break;
        }

        let poll_mask = if sm.is_dragging() { 31 } else { 15 };
        if (tick & poll_mask) == 0 && session.poll_should_quit() {
            break;
        }
        tick = tick.wrapping_add(1);

        let frame = session.sample_inputs(sm.active_device_hint());
        match sm.step_with(frame, || session.get_working_standing_pose()) {
            StepAction::UpdatePlayspace(m) => {
                last_drag_release_ns = None;
                session.apply_working_standing_pose(&m);
            }
            StepAction::CommitPlayspace(_m) => {
                last_drag_release_ns = Some(clock.now_ns());
            }
            StepAction::Idle | StepAction::AbortDrag => {
                if !sm.is_dragging() {
                    if let Some(t_rel) = last_drag_release_ns {
                        if clock.now_ns().saturating_sub(t_rel) >= 3_000_000_000 {
                            session.flush_pending_commit();
                            last_drag_release_ns = None;
                        }
                    }
                }
            }
        }
    }

    if let DragPhase::Dragging {
        moved: true,
        last_m,
        ..
    } = sm.phase()
    {
        let _ = session.commit_working_standing_pose(&last_m);
    } else {
        session.flush_pending_commit();
    }

    0
}

fn app_main() -> i32 {
    let simd = SimdMath::new_detected();

    if command_line_has_flag(b"--help") || command_line_has_flag(b"-h") {
        print_out(format_args!(
            "nexashift - headless reactive 1:1 3D SteamVR playspace drag (simd={})\n\
             usage: nexashift [--poc | --bench | --help]\n",
            simd.level_name()
        ));
        return 0;
    }
    if command_line_has_flag(b"--poc") {
        return run_poc(simd);
    }
    if command_line_has_flag(b"--bench") {
        return run_bench(simd);
    }
    run_daemon(simd)
}

#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    app_main()
}

#[cfg(test)]
fn main() {
    let _ = app_main();
}
