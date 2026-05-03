# Characterize Project State

## Project
Characterize turns images into character art and can play videos as character art in the terminal or a pygame window.

## Current State
- CLI entry point is `characterize.py`.
- Video playback is implemented by `charlib/player.py`.
- Terminal video audio uses `ffplay` directly.
- Windowed video audio uses `pygame.mixer` after extracting a temporary WAV with `ffmpeg`.
- Video playback supports `--no-audio`.
- CLI numeric options are validated before work starts.
- Text output writes UTF-8 character rows without inserting spaces between characters.
- Smoke coverage lives in `smoke_test.py`.

## Commands
- Install dependencies: `pip install -r requirements.txt`
- Syntax check: `python -m py_compile charlib\player.py characterize.py`
- Smoke test: `python smoke_test.py`
- Terminal video: `python characterize.py -i path\to\video.mp4 --video --terminal -W 80 --color`

## Notes
- Full terminal audio support requires `ffplay` on PATH.
- Windowed playback requires `pygame`.
- FileOptimizer integration calls the executable directly instead of through a shell command.
- PowerShell execution-policy profile warnings are environment noise; run commands with `powershell -NoProfile` to suppress them.
