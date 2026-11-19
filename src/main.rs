#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

use core::fmt::Write;

use nexashift::math::{HmdMatrix34_t, SimdMath, Vec3};
use nexashift::openvr::{OpenVrError, OpenVrSession, ShmMonitorManager, set_steamvr_autolaunch};
use nexashift::platform::{
    MonotonicClock, ReactiveWaiter, StackBuf, WaitOutcome, command_line_has_flag,
    install_signal_handlers, is_running, query_chaperone_file_stamp, query_process_stats,
    write_stderr, write_stdout,
};
use nexashift::state::{
    DragStateMachine, HandInput, InputFrame, PersistOp, PlayspacePersistenceController, StepAction,
};

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
        OpenVrError::AppInstallFailed(stage, code) => {
            print_err(format_args!(
                "error: IVRApplications::{} failed (EVRApplicationError={})\n",
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

#[inline]
fn idle_period_ms_for_hmd(hz: f32) -> u32 {
    let raw = (1000.0f32 / hz) + 0.5;
    let ms = raw as u32;
    ms.clamp(8, 12)
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
    session.restore_working_preview();
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

    let _ = session.sample_inputs(None, None, 0);
    let clock = MonotonicClock::new();
    let mut shm_mgr = ShmMonitorManager::new(&session, clock.now_ns());
    let idle_period_ms = idle_period_ms_for_hmd(session.hmd_refresh_hz());

    print_out(format_args!(
        "bench[init]: simd={}, runtime={}, hmd_hz={}, shm={}, layout={}, trust={}, shm_slots={}, bound_slots={}, idle_period_ms={}\n",
        simd.level_name(),
        session.runtime_version(),
        session.hmd_refresh_hz() as u32,
        shm_mgr.is_shm_active(),
        shm_mgr.layout_name(),
        shm_mgr.trust().as_str(),
        shm_mgr.watched_slot_count(),
        shm_mgr.watched_bound_count(),
        idle_period_ms
    ));

    let waiter = ReactiveWaiter::new();
    let mut sm = DragStateMachine::with_simd(simd);

    // 1. Bare-metal idle benchmark (5 seconds at native VR frame rate via high-res timer + SIMD SHM watch).
    let bench_duration_ns: u64 = 5_000_000_000;
    let idle_chap_before = query_chaperone_file_stamp();
    let idle_start_stats = query_process_stats();
    let idle_t0 = clock.now_ns();
    let mut idle_wakeups: u64 = 0;

    while is_running() {
        let now_ns = clock.now_ns();
        if now_ns.saturating_sub(idle_t0) >= bench_duration_ns {
            break;
        }
        let watch_handle = shm_mgr.vrserver_wait_handle();
        match waiter.wait_idle_tick(watch_handle, idle_period_ms) {
            WaitOutcome::Tick => {}
            WaitOutcome::Stop | WaitOutcome::ServerExited => break,
        }
        if (idle_wakeups & 63) == 0 && session.poll_should_quit() {
            session.acknowledge_quit_exiting();
            break;
        }
        let now_ns = clock.now_ns();
        if shm_mgr.poll_idle(&session, simd, now_ns) {
            let frame = session.sample_inputs(
                sm.active_device_hint(),
                shm_mgr.monitor_mut(),
                now_ns,
            );
            let _ = sm.step_with(frame, || session.get_working_standing_pose());
        }
        idle_wakeups += 1;
    }

    waiter.disarm_idle_timer();
    let idle_wall_ns = clock.now_ns().saturating_sub(idle_t0).max(1);
    let idle_end_stats = query_process_stats();
    let idle_chap_after = query_chaperone_file_stamp();
    let idle_chap_modified = idle_chap_before != idle_chap_after;
    let idle_cpu_ns = idle_end_stats
        .cpu_time_ns
        .saturating_sub(idle_start_stats.cpu_time_ns);
    let idle_cycles = idle_end_stats
        .cpu_cycles
        .saturating_sub(idle_start_stats.cpu_cycles);
    let idle_mcyc_s = (idle_cycles as f64) * 1_000.0 / (idle_wall_ns as f64);
    let idle_cpu_pct_1core = (idle_cpu_ns as f64) * 100.0 / (idle_wall_ns as f64);
    let idle_cpu_pct_12t = idle_cpu_pct_1core / 12.0;
    let idle_wakeups_sec = (idle_wakeups as f64) * 1_000_000_000.0 / (idle_wall_ns as f64);
    let idle_ipc_writes = idle_end_stats
        .write_bytes
        .saturating_sub(idle_start_stats.write_bytes);

    print_out(format_args!(
        "bench[idle]: wall_ms={:.1}, wakeups={} ({:.1}/s), Mcycles/s={:.2}, cpu_ms={:.3}, cpu_1core={:.4}%, cpu_12t={:.4}%, private_kb={}, ipc_write_bytes={}, chaperone_modified={}\n",
        idle_wall_ns as f64 / 1_000_000.0,
        idle_wakeups,
        idle_wakeups_sec,
        idle_mcyc_s,
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
        if !is_running() {
            break;
        }
        if (drag_wakeups & 31) == 0 && session.poll_should_quit() {
            session.acknowledge_quit_exiting();
            break;
        }
        let iter_t0 = clock.now_ns();
        let _real_frame =
            session.sample_inputs(drag_sm.active_device_hint(), shm_mgr.monitor_mut(), iter_t0);
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
    session.restore_working_preview();
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

fn run_probe(simd: SimdMath) -> i32 {
    let session = match OpenVrSession::init(true) {
        Ok(s) => s,
        Err(e) => {
            print_openvr_error(e);
            return 1;
        }
    };

    let _ = session.sample_inputs(None, None, 0);
    let clock = MonotonicClock::new();
    let shm_mgr = ShmMonitorManager::new(&session, clock.now_ns());

    print_out(format_args!(
        "probe: simd={}, runtime={}, hmd_hz={}, shm={}, layout={}, trust={}, shm_slots={}, bound_slots={}, fast_drag={}\n",
        simd.level_name(),
        session.runtime_version(),
        session.hmd_refresh_hz() as u32,
        shm_mgr.is_shm_active(),
        shm_mgr.layout_name(),
        shm_mgr.trust().as_str(),
        shm_mgr.watched_slot_count(),
        shm_mgr.watched_bound_count(),
        shm_mgr.has_validated_bindings()
    ));

    0
}

#[inline]
fn handle_step_action(
    session: &OpenVrSession,
    persistence: &mut PlayspacePersistenceController,
    action: StepAction,
) {
    match action {
        StepAction::Idle => {}
        StepAction::UpdatePlayspace(m) => {
            persistence.on_preview_update(m);
            session.apply_working_standing_pose(&m);
        }
        StepAction::CommitPlayspace(m) => {
            if let PersistOp::CommitNow(c) =
                persistence.on_drag_finished(m, session.is_dashboard_visible())
            {
                session.commit_working_standing_pose(&c);
            }
        }
        StepAction::AbortDrag => {
            if let PersistOp::CommitNow(c) =
                persistence.on_drag_aborted(session.is_dashboard_visible())
            {
                session.commit_working_standing_pose(&c);
            }
        }
    }
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

    let startup_pose = session.get_working_standing_pose();
    let _ = session.sample_inputs(None, None, 0);
    let clock = MonotonicClock::new();
    let mut shm_mgr = ShmMonitorManager::new(&session, clock.now_ns());
    let idle_period_ms = idle_period_ms_for_hmd(session.hmd_refresh_hz());

    let waiter = ReactiveWaiter::new();
    let mut sm = DragStateMachine::with_simd(simd);
    let mut persistence = PlayspacePersistenceController::new(startup_pose);
    let mut tick: u64 = 0;
    let mut idle_timer_armed = false;
    let mut steamvr_quitting = false;

    while is_running() {
        let was_dragging = sm.is_dragging();
        if was_dragging {
            if idle_timer_armed {
                waiter.disarm_idle_timer();
                idle_timer_armed = false;
            }
            wait_next_vr_frame(&session, &waiter);
            if !is_running() {
                break;
            }
            if (tick & 31) == 0 && session.poll_should_quit() {
                steamvr_quitting = true;
                break;
            }
            tick = tick.wrapping_add(1);

            let now_ns = clock.now_ns();
            let frame =
                session.sample_inputs(sm.active_device_hint(), shm_mgr.monitor_mut(), now_ns);
            let action = sm.step_with(frame, || session.get_working_standing_pose());
            handle_step_action(&session, &mut persistence, action);
            if !sm.is_dragging() {
                shm_mgr.on_drag_finished(&session, clock.now_ns());
            }
            continue;
        }

        let watch_handle = shm_mgr.vrserver_wait_handle();
        idle_timer_armed = true;
        match waiter.wait_idle_tick(watch_handle, idle_period_ms) {
            WaitOutcome::Tick => {}
            WaitOutcome::Stop | WaitOutcome::ServerExited => break,
        }
        if (tick & 63) == 0 && session.poll_should_quit() {
            steamvr_quitting = true;
            break;
        }
        tick = tick.wrapping_add(1);

        if persistence.has_pending_commit()
            && let PersistOp::CommitNow(c) =
                persistence.poll_deferred_commit(session.is_dashboard_visible())
        {
            session.commit_working_standing_pose(&c);
        }

        let now_ns = clock.now_ns();
        if !shm_mgr.poll_idle(&session, simd, now_ns) {
            continue;
        }

        let frame = session.sample_inputs(
            sm.active_device_hint(),
            shm_mgr.monitor_mut(),
            now_ns,
        );
        let action = sm.step_with(frame, || session.get_working_standing_pose());
        handle_step_action(&session, &mut persistence, action);
    }

    if idle_timer_armed {
        waiter.disarm_idle_timer();
    }
    if steamvr_quitting {
        if let PersistOp::CommitNow(m) = persistence.on_steamvr_quit() {
            session.commit_working_standing_pose(&m);
        }
        session.acknowledge_quit_exiting();
    } else if let PersistOp::CommitNow(m) = persistence.on_app_exit() {
        session.commit_working_standing_pose(&m);
    }

    0
}

fn app_main() -> i32 {
    let simd = SimdMath::new_detected();

    if command_line_has_flag(b"--help") || command_line_has_flag(b"-h") {
        print_out(format_args!(
            "nexashift - headless reactive 1:1 3D SteamVR playspace drag (simd={})\n\
             usage: nexashift [--install | --uninstall | --probe | --poc | --bench | --help]\n",
            simd.level_name()
        ));
        return 0;
    }
    if command_line_has_flag(b"--install") {
        return match set_steamvr_autolaunch(true) {
            Ok(()) => {
                print_out(format_args!(
                    "nexashift: registered with SteamVR auto-launch (nexashift.spacedrag)\n"
                ));
                0
            }
            Err(e) => {
                print_openvr_error(e);
                1
            }
        };
    }
    if command_line_has_flag(b"--uninstall") {
        return match set_steamvr_autolaunch(false) {
            Ok(()) => {
                print_out(format_args!(
                    "nexashift: removed from SteamVR auto-launch\n"
                ));
                0
            }
            Err(e) => {
                print_openvr_error(e);
                1
            }
        };
    }
    if command_line_has_flag(b"--probe") {
        return run_probe(simd);
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
