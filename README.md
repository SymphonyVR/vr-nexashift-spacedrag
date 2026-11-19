# NexaShift

A lightweight, headless `#![no_std]` Rust utility for 1:1 3D SteamVR playspace dragging (`space_drag`), designed for **~0.0% idle CPU usage** and native headset-frame responsiveness both in-game and inside the SteamVR Dashboard.

## Highlights

- **1:1 Playspace Dragging:** Grab any point in 3D space with a controller and move your standing playspace without cumulative floating-point drift.
- **Works In-Game & in the SteamVR Dashboard:** Normal SteamVR background apps stop receiving controller actions whenever the Dashboard opens. NexaShift automatically falls back to reading your bound controller buttons from SteamVR's shared memory so you can drag anywhere.
- **~0.0% Idle CPU at Headset Refresh Rate:** Uses a high-resolution OS timer (every `8–12 ms`, matching `90–144 Hz` headsets) and checks only a few bytes of read-only shared memory with runtime-detected SIMD (`fearless_simd`), skipping SteamVR's heavier input API until a button actually changes.
- **Session-Only (Never Overwrites Room Setup):** Moves stay in SteamVR's live working-set preview in memory while you play and revert cleanly when NexaShift exits—your calibrated room center and `chaperone_info.vrchap` file are never touched, so SteamVR always starts centered next time and never resets your Dashboard mid-session.
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

### Auto-Start with SteamVR

While SteamVR is running, you can register NexaShift to launch automatically whenever SteamVR starts:

```powershell
# Register manifest/nexashift.vrmanifest and enable SteamVR auto-launch
.\target\release\nexashift.exe --install

# Remove NexaShift from SteamVR auto-launch
.\target\release\nexashift.exe --uninstall
```

You can also toggle it anytime inside SteamVR under **Settings → Startup/Shutdown → Choose Startup Overlay Apps** (`NexaShift`).

> **Note:** [manifest/nexashift.vrmanifest](manifest/nexashift.vrmanifest) points to `../target/release/nexashift.exe` by default. If you move the executable elsewhere, keep `manifest/` next to it and update `binary_path_windows` (or `binary_path_linux`) inside `nexashift.vrmanifest` before running `--install`.

### CLI Modes

```text
nexashift [--install | --uninstall | --probe | --poc | --bench | --help]
```

- *(no flags)* — Starts the background daemon attached to the running SteamVR session.
- `--install` / `--uninstall` — Registers or removes NexaShift from SteamVR's startup overlay apps (`manifest/nexashift.vrmanifest`).
- `--probe` — Read-only diagnostic check reporting OpenVR runtime version, validated/discovered shared-memory layout, trust status, and watched slot counts without moving your playspace.
- `--poc` — Runs a 120-frame live movement test in memory, verifies readback accuracy, and restores your original playspace.
- `--bench` — Runs a 5-second idle benchmark followed by a 5-second simulated drag benchmark, reporting wakeup rate, CPU cycles/time, memory usage, and latency.
- `--help`, `-h` — Prints CLI usage and the runtime-detected SIMD level.

## How It Works

1. **Validated Shared-Memory Discovery & Recovery:** Instead of relying blindly on hardcoded SteamVR internal offsets, NexaShift validates known layout descriptors (`VirtualQuery` region size/alignment checks and OpenVR handle anchors) at startup and automatically probes shifted layouts or recovers (`5s -> 15s -> 60s` backoff with periodic 60-second cross-checks) if SteamVR updates or reloads bindings.
2. **Idle Watching:** NexaShift maps SteamVR's read-only shared memory only for your bound Space Drag buttons and wakes on a high-resolution timer (`8–12 ms`), skipping `UpdateActionState`/`GetDigitalActionData` calls until a button actually changes.
3. **Active Dragging:** Once you press a drag button, the loop switches to compositor frame sync (`WaitFrameSync`), evaluates validated shared-memory bindings directly alongside live controller tracking poses, and applies `1:1` inverse motion relative to your starting anchor pose—so there is zero frame-to-frame drift.
4. **Dashboard Fallback & OpenVR Safety:** When SteamVR marks action data as inactive while the Dashboard is open, NexaShift continues evaluating the validated shared-memory button slots (including the 500 ms double-press detector for `B`/`Y` or Menu buttons). If a future SteamVR build changes shared memory beyond validated discovery, NexaShift falls back to public OpenVR input rather than guessing offsets.
5. **Session-Only Position (No Disk Saves):** Live updates use SteamVR's in-memory working-set preview (`ShowWorkingSetPreview`) and never call `CommitWorkingCopy`. This keeps your saved room calibration intact for your next SteamVR launch and ensures `vrcompositor` never receives room-setup commit events that reset the Dashboard mid-session.

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
