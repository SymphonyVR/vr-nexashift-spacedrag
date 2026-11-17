# NexaShift

A lightweight, background SteamVR app that lets you grab and pull your VR playspace in 3D space (1:1 movement) while using **~0.0% CPU** when idle.

No window, no overlay clutter — just start it alongside SteamVR and drag.

## Features

- **1:1 Playspace Dragging:** Hold the drag button and move your controller to pull the world around you smoothly in any direction. Walking around your room doesn't drift or fight your drag.
- **Works Everywhere in VR (In-Game & SteamVR Dashboard):** Keeps working even when the SteamVR Dashboard or overlay menus are open, where SteamVR normally blocks background apps from seeing controller buttons.
- **Instant Response, ~0.0% Idle CPU:** Checks your controllers at your headset's refresh rate (around every 8–12 ms) without waking up SteamVR's heavy input system until you actually press a button.
- **Gentle on Your Drive:** Moves your playspace smoothly in memory while you drag, then waits **3 seconds after you let go** (or when exiting) before saving the new position to SteamVR's settings file.

## Default Controls

You can customize these anytime inside SteamVR's **Controller Bindings** settings:

| Headset / Controllers | Hold to Drag | Alternate (Double-Press & Hold) |
| --- | --- | --- |
| **Oculus Touch / Pico / Quest** | Left or Right **Joystick Click** | Left **Y** or Right **B** |
| **Valve Index (Knuckles)** | Left or Right **Thumbstick Click** | Left or Right **B** |
| **HTC Vive Wands** | Left or Right **Trackpad Click** | Left or Right **Menu Button** |
| **Windows Mixed Reality** | Left or Right **Joystick Click** | Left or Right **Menu Button** |

## Quick Start

### Requirements

- **SteamVR** running (built and tested on Windows; Linux support is also included).
- **Rust** toolchain (`cargo`) if building from source.

### Build & Run

```powershell
# Build the release binary
cargo build --release

# Start NexaShift in the background
.\target\release\nexashift.exe
```

Keep the `manifest/` folder alongside the executable (or run from the project folder) so SteamVR can find the action and default controller bindings.

### Extra Command-Line Options

```text
nexashift [--poc | --bench | --help]
```

- *(no flags)* — Normal mode: runs quietly in the background with SteamVR.
- `--poc` — Runs a short automated 120-frame movement test and restores your playspace back where it started.
- `--bench` — Runs a 10-second performance test (5s idle + 5s simulated dragging) and prints CPU, memory, and latency numbers.
- `--help`, `-h` — Shows command usage.

## Tests

```powershell
cargo test
cargo check --target x86_64-unknown-linux-gnu
```

<details>
<summary>Technical Details (Under the Hood)</summary>

- **No Standard Library (`#![no_std]`):** Compiles with zero heap allocations (`no_std` / no `alloc`) for minimal memory footprint (`~5.6 MB` private memory) and deterministic timing.
- **Low-Overhead Idle Watching:** Reads SteamVR's shared memory (`VR_InputState` and per-controller `PC_<container>_storage0` component slots) via read-only memory maps and runtime-detected SIMD (`fearless_simd`), synchronized with a high-resolution Windows waitable timer (`CreateWaitableTimerExW(..., CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, ...)`). SteamVR's `UpdateActionState` mutex/allocations are only touched when a bound input actually changes or during an active drag.
- **Dashboard Input Fallback:** Resolves active bindings via `IVRInput::GetActionBindingInfo` and `GetInputSourceHandle` down to their raw shared-memory component offsets, with a `#![no_std]` double-press detector (`500 ms` window), so dragging works even when SteamVR suppresses `GetDigitalActionData` (`b_active == false`) inside the Dashboard.
- **Safe Chaperone Commits:** Uses `IVRChaperoneSetup::ShowWorkingSetPreview()` during drags synced via `IVROverlay::WaitFrameSync()`, and debounces `CommitWorkingCopy()` by 3 seconds after release.

### Project Structure

- [src/main.rs](src/main.rs) — Entry point, `--poc` / `--bench` tests, and main loop.
- [src/math.rs](src/math.rs) — 3D transforms and runtime-detected `fearless_simd` math helpers.
- [src/state.rs](src/state.rs) — Drag state machine (single-hand lock, tracking-loss protection, zero drift).
- [src/openvr.rs](src/openvr.rs) — Raw OpenVR FFI bindings and shared-memory input monitor.
- [src/platform.rs](src/platform.rs) — Windows and Linux low-level OS helpers (timers, shared memory, clocks).
- [manifest/](manifest/) — SteamVR action manifest and default controller profiles.

</details>

## License

[MIT](LICENSE)
