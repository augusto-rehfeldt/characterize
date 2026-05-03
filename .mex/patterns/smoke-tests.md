# Smoke Tests

Use `python smoke_test.py` for a quick local health check. The script creates temporary fixtures under `.tmp_smoke`, runs syntax checks, verifies CLI validation, previews a generated image in the terminal, exports text output, and runs a generated terminal video with `--no-audio` when `ffmpeg` is available.

The script deletes `.tmp_smoke` and its generated text export before exiting.
