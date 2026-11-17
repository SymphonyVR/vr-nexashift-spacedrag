# NexaShift

A lightweight, headless `#![no_std]` Rust utility for 1:1 3D SteamVR playspace dragging (`space_drag`), designed for **~0.0% idle CPU usage** and native headset-frame responsiveness both in-game and inside the SteamVR Dashboard.

## Highlights

- **1:1 Playspace Dragging:** Grab any point in 3D space with a controller and move your standing playspace without cumulative floating-point drift.
- **Works In-Game & in the SteamVR Dashboard:** Normal SteamVR background apps stop receiving controller actions whenever the Dashboard opens. NexaShift automatically falls back to reading your bound controller buttons from SteamVR's shared memory so you can drag anywhere.
- **~0.0% Idle CPU at Headset Refresh Rate:** Uses a high-resolution OS timer (every `8–12 ms`, matching `90–144 Hz` headsets) and checks only a few bytes of read-only shared memory with runtime-detected SIMD (`fearless_simd`), skipping SteamVR's heavier input API until a button actually changes.
- **Debounced Disk Saves:** During a drag, changes stay in memory (previewed every VR frame). SteamVR's `chaperone_info.vrchap` config file is only written to disk once, 3 seconds after you release the drag (or when exiting).
- **Zero Heap Allocations (`#![no_std]`):** Written in pure `#![no_std]` Rust without `alloc`, calling OpenVR and OS APIs directly via C FFI.

## Default Controls

Defined in [manifest/actions.json](manifest/actions.json) and editable anytime in SteamVR's **Controller Bindings** UI:

| Controller Profile | Click & Hold | Double-Press & Hold |
| --- | --- | --- |
| **Oculus Touch / Pico / Quest** | Left or Right **Joystick Click** | Left **Y** or Right **B** |
| **Valve Index (Knuckles)** | Left or Right **Thumbstick Click** | Left or Right **B** |
| **HTC Vive Wands** | Left or Right **Trackpad Click** | Left or Right **Menu Button** |
| **Windows Mixed Reality** | Left or Right **Joystick Click** | Left or Right **Menu Button** |

## Quick Start

### Requirements

- **SteamVR** running (tested on Windows; Linux OS bindings are also included in [src/platform.rs](src/platform.rs)).
- **Rust** toolchain (`cargo`, edition 2024).

### Build & Run

```powershell
# Build optimized release binary
cargo build --release

# Start the background daemon
.\target\release\nexashift.exe
```

Keep the [manifest/](manifest/) directory accessible from the working directory (or alongside the executable) so SteamVR can load `manifest/actions.json` and the default binding files.

### CLI Modes

```text
nexashift [--poc | --bench | --help]
```

- *(no flags)* — Starts the background daemon attached to the running SteamVR session.
- `--poc` — Runs a 120-frame live movement test in memory, verifies readback accuracy, and restores your original playspace.
- `--bench` — Runs a 5-second idle benchmark followed by a 5-second simulated drag benchmark, reporting wakeup rate, CPU cycles/time, memory usage, and latency.
- `--help`, `-h` — Prints CLI usage and the runtime-detected SIMD level.

## How It Works

1. **Idle Watching:** Instead of polling OpenVR's full action system every tick (which takes locks and allocates memory inside SteamVR's client DLL), NexaShift maps SteamVR's read-only shared memory for your bound drag buttons and wakes on a high-resolution timer (`8–12 ms`).
2. **Active Dragging:** Once you press a drag button, the loop switches to compositor frame sync (`WaitFrameSync`), captures your controller's starting pose and playspace matrix, and applies `1:1` inverse motion relative to that anchor pose—so there is zero frame-to-frame drift.
3. **Dashboard Fallback:** If SteamVR marks action data as inactive while the Dashboard is open, NexaShift evaluates the bound raw button slots from shared memory directly (including a 500 ms double-press state machine for `B`/`Y` or Menu buttons) alongside live controller tracking poses.
4. **Saving Position:** Live updates use SteamVR's working-set preview in memory; `CommitWorkingCopy` is deferred until 3 seconds after drag release or shutdown.

## Tests

```powershell
cargo test
cargo check --target x86_64-unknown-linux-gnu
```

## Project Structure

- [src/main.rs](src/main.rs) — `#![no_std]` entry point, `--poc` / `--bench` runners, and daemon event loop.
- [src/state.rs](src/state.rs) — Pure state machine (`DragStateMachine`) handling anchor capture, single-hand lock, and tracking-loss recovery.
- [src/math.rs](src/math.rs) — 3D rigid transform matrices (`HmdMatrix34_t`) and runtime-detected `fearless_simd` math helpers.
- [src/openvr.rs](src/openvr.rs) — Raw OpenVR C ABI vtable structs, shared-memory input watcher, and binding resolver.
- [src/platform.rs](src/platform.rs) — Windows & Linux `#![no_std]` OS layer (DLL loading, high-res timers, shared memory, monotonic clock, process stats).
- [manifest/](manifest/) — OpenVR action manifest, `.vrmanifest`, and default bindings for supported controllers.

## License

[MIT](LICENSE)
