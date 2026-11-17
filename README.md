# NexaShift

A headless `#![no_std]` Rust utility for 1:1 3D OpenVR / SteamVR playspace dragging (`space_drag`), engineered for near-zero idle CPU usage and native VR-frame responsiveness both in-game and while the SteamVR Dashboard is open.

## Features

- **1:1 3D Playspace Dragging:** Translates the standing playspace directly from raw controller deltas relative to the captured anchor pose, avoiding incremental drift and canceling out roomscale walking when a controller is stationary relative to the body.
- **Works In-Game and in the SteamVR Dashboard:**
  - Uses `IVRInput` (`GetDigitalActionData`) when SteamVR delivers action input normally.
  - Automatically falls back to resolving the bound `VR_InputState` / per-controller property container (`PC_<container>_storage0`) shared-memory components when SteamVR suppresses background action sets (such as when the SteamVR Dashboard is open), including `#![no_std]` `double`-press detection.
- **0.0% Idle CPU with Native Frame Latency:**
  - Wakes at native HMD cadence (`8–12 ms`, clamped from `1000 / HMD_Hz`) using a high-resolution waitable timer (`CreateWaitableTimerExW(..., CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, ...)` on Windows) rather than coarse `15.625 ms` scheduler ticks.
  - Watches only the resolved bound input component slots in read-only shared memory with runtime-detected `fearless_simd` (e.g., AVX2 / SSE / NEON / scalar fallback), avoiding per-tick `VR_InputMutex` calls and `vrclient` heap allocations while idle.
  - Switches to compositor frame synchronization (`IVROverlay::WaitFrameSync`) during active drags and debounces `CommitWorkingCopy` (`3s` after drag release or on shutdown) to avoid chaperone disk churn.
- **Strictly `#![no_std]` Core:** Zero heap allocation (`no_std` / no `alloc`) in the binary and library.

## Default Bindings

Defined in [manifest/actions.json](manifest/actions.json) (`/actions/main/in/space_drag`) and customizable through the SteamVR Controller Binding UI:

| Controller Type | Primary Drag (Click & Hold) | Alternate Drag (Double-Press & Hold) |
| --- | --- | --- |
| `oculus_touch` (incl. Pico) | Left / Right Joystick Click | Left **Y** / Right **B** Double-Press |
| `knuckles` (Valve Index) | Left / Right Thumbstick Click | Left / Right **B** Double-Press |
| `vive_controller` | Left / Right Trackpad Click | Left / Right Application Menu Double-Press |
| `holographic_controller` (WMR) | Left / Right Joystick Click | Left / Right Application Menu Double-Press |

## Requirements

- **SteamVR** running (tested on Windows; Linux `#![no_std]` OpenVR/POSIX layer is included in [src/platform.rs](src/platform.rs)).
- **Rust toolchain** (edition 2024).

## Build & Run

```powershell
# Build release binary
cargo build --release

# Run the headless playspace-drag daemon
.\target\release\nexashift.exe
```

Make sure the `manifest/` directory stays alongside the executable (or in the workspace root when running from `target/release/`) so SteamVR can load `manifest/actions.json` and the default binding JSON files.

### CLI Modes

```text
nexashift [--poc | --bench | --help]
```

- *(no flags)* — Starts the headless daemon (`run_daemon`) and attaches to the active SteamVR session.
- `--poc` — Runs a 120-frame live working-standing-pose preview excursion and verifies readback, restoration, and unchanged chaperone config.
- `--bench` — Measures idle wakeups, CPU cycles/time, private memory, IPC writes, and active drag latency over 5-second idle and drag windows.
- `--help`, `-h` — Prints CLI usage and detected runtime SIMD level.

## Tests

```powershell
cargo test
cargo check --target x86_64-unknown-linux-gnu
```

## Project Structure

- [src/main.rs](src/main.rs) — `#![no_std]` entry point, `--poc` / `--bench` harnesses, and reactive daemon loop.
- [src/math.rs](src/math.rs) — Rigid transforms (`HmdMatrix34_t`), 3D vectors, and runtime-detected `fearless_simd` helpers.
- [src/state.rs](src/state.rs) — Deterministic `DragStateMachine` (`Idle`, `Dragging`, tracking-loss safety, single-hand lock).
- [src/openvr.rs](src/openvr.rs) — Raw OpenVR ABI FFI tables (`IVRSystem`, `IVRChaperoneSetup`, `IVRInput`, `IVROverlay`, `IVRSettings`) and `BareMetalInputMonitor`.
- [src/platform.rs](src/platform.rs) — Windows & Linux `#![no_std]` dynamic loading, high-res timers, shared-memory mapping, and clock/process stats.
- [manifest/](manifest/) — OpenVR action manifest, `.vrmanifest`, and controller default bindings.

## License

[MIT](LICENSE)
