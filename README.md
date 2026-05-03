# Characterize

Characterize turns images into character art and can also play video as ASCII in the terminal or in a window.

## Entry Point

Use `characterize.py` as the main command:

```bash
python characterize.py [arguments]
```

`characterize_video.py` remains as a compatibility wrapper for older scripts. It forwards to `characterize.py --video`.

## Install

```bash
pip install -r requirements.txt
```

Run smoke checks:

```bash
python smoke_test.py
```

Video playback works best with `ffmpeg` installed. Terminal audio also needs `ffplay`, which is usually included with full ffmpeg installs:

```bash
ffmpeg -version
ffplay -version
```

If `ffmpeg` or `ffplay` is missing, Characterize can still play video, but audio support will be limited.

## Modes

Characterize now uses one CLI with a few clear switches:

`--terminal`
: Render output directly in the terminal.

`--video`
: Force video playback mode.

`--no-audio`
: Disable video audio playback.

Auto mode:
: When the input is a still image, Characterize previews it in the terminal automatically.

## Quick Start

Preview a single image in the terminal:

```bash
python characterize.py -i path/to/image.png --terminal -W 80
```

Export a folder of images to output files:

```bash
python characterize.py -i path/to/folder -f png,txt -l ascii -C 12
```

Play a video in the terminal:

```bash
python characterize.py -i path/to/video.mp4 --video --terminal -W 80 --color
```

Play a video in the terminal without audio:

```bash
python characterize.py -i path/to/video.mp4 --video --terminal --no-audio -W 80 --color
```

Play a video in a window:

```bash
python characterize.py -i path/to/video.mp4 --video -W 100 -H 45
```

Use the compatibility wrapper if an older script still calls it:

```bash
python characterize_video.py -i path/to/video.mp4 --terminal
```

## Video UI

Windowed video mode includes playback controls at the bottom of the window:

`Space`
: Pause or resume playback.

`Left` / `Right`
: Seek backward or forward by 5 seconds.

`+` / `-`
: Adjust volume.

`F`
: Toggle fullscreen.

`1` / `2` / `3`
: Switch between preset window sizes.

`C`
: Toggle color rendering.

`T`
: Toggle true-color rendering.

## Troubleshooting

PowerShell may print an execution-policy warning if your user profile script is blocked. Characterize does not need that profile script; run project commands from `powershell -NoProfile` or update the local PowerShell execution policy if you want to remove the warning.

The timeline shows current time, total time, and playback progress. You can click or drag on the timeline to seek.

## Arguments

`-i`, `--input`
: Input files or folders. Multiple values are allowed.

`-W`, `--width`
: Output width in characters.

`-H`, `--height`
: Output height in characters. If omitted, height is derived from the input aspect ratio.

`-l`, `--language`
: Character set to use.

`-C`, `--complexity`
: Number of characters to keep in the palette.

`--empty`
: Include a blank character for darker areas.

`-c`, `--color`
: Enable color output.

`--true-color`
: Use actual pixel colors instead of palette-based brightness sampling.

`--terminal`
: Render in the terminal instead of the windowed player.

`--video`
: Force video mode.

`-f`, `--format`
: Image output format for image mode. Supported values: `png`, `jpg`, `txt`.

`-d`, `--divide`
: Subdivide large images before processing.

`-o`, `--optimize`
: Optimize generated image files when possible.

`--recursive`
: Scan input folders recursively.

## Output

Image mode writes files under:

```text
output/<language>/
```

Text output is written alongside the image outputs when `txt` is selected.

## Examples

High-contrast terminal preview:

```bash
python characterize.py -i poster.png --terminal -W 120 -l braille -C 16 --color
```

Batch export with text and PNG:

```bash
python characterize.py -i ./photos -f png,txt -l ascii -C 12 --recursive
```

Video playback with true color:

```bash
python characterize.py -i concert.mp4 --video --terminal -W 100 --true-color
```

If you want the same command to work on both an image and a video, just pass the input and let Characterize choose the mode. Still images preview in the terminal, and video uses the player unless you override it.
