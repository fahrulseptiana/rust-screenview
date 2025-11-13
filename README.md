# screenview

screenview is a Windows-only desktop viewer that lets you mirror any connected display in a dedicated window with minimal latency. Frames are captured via the Windows Graphics Capture API (including the system cursor) and rendered directly on the GPU through `wgpu`, so the preview stays sharp while keeping CPU usage low.

## Features

- Lists available monitors and prompts you to pick one to mirror.
- Opens maximized by default, supports double-click to toggle fullscreen, and ESC to exit fullscreen.
- Captures the cursor and video stream using `windows-capture`.
- GPU-driven rendering with aspect-ratio preservation so the preview fits the window without distortion.
- Configurable FPS via CLI flag (`--fps`, defaults to 60).

## Requirements

- Windows 10 or later with Graphics Capture support.
- Rust toolchain (1.76+ recommended).

## Usage

```powershell
cargo run --release -- [OPTIONS]
```

Common options:

- `--display <index>` – zero-based display index; if omitted you’ll be prompted.
- `--fps <value>` – capture FPS (1–240), default 60.

Example:

```powershell
cargo run --release -- --display 1 --fps 75
```

## Controls

- **Double-click** anywhere in the window to toggle fullscreen.
- **ESC** exits fullscreen (or closes the app if already windowed).
- Closing the window stops the capture thread and exits.

## Building

```powershell
cargo fmt
cargo check
cargo build --release
```

The release binary will be available at `target/release/screenview.exe`.
