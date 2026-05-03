import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent


def run_command(command, timeout=30):
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def assert_ok(result, label):
    if result.returncode != 0:
        print(f"{label} failed with exit code {result.returncode}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise SystemExit(result.returncode)


def make_image(path):
    image = Image.new("RGB", (32, 18), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 14, 15), fill=(220, 220, 220))
    draw.ellipse((17, 3, 30, 16), fill=(120, 180, 255))
    image.save(path)


def make_video(path):
    if not shutil.which("ffmpeg"):
        print("Skipping video smoke: ffmpeg is not on PATH.")
        return False

    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x18:rate=4:duration=0.5",
            "-an",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ]
    )
    assert_ok(result, "ffmpeg fixture generation")
    return True


def main():
    unique = f"characterize_smoke_{uuid.uuid4().hex}"
    output_text = ROOT / "output" / "ascii" / f"{unique}.txt"

    temp_path = ROOT / ".tmp_smoke"
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir()

    try:
        image_path = temp_path / f"{unique}.png"
        video_path = temp_path / f"{unique}.mp4"
        make_image(image_path)

        syntax = run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                "characterize.py",
                "charlib/player.py",
                "charlib/processing.py",
                "charlib/file_utils.py",
            ]
        )
        assert_ok(syntax, "syntax check")

        invalid_width = run_command(
            [sys.executable, "characterize.py", "--width", "0", "-i", str(image_path)]
        )
        if invalid_width.returncode == 0 or "--width must be a positive integer" not in invalid_width.stderr:
            print("argument validation smoke failed")
            print(invalid_width.stdout)
            print(invalid_width.stderr)
            raise SystemExit(1)

        terminal_preview = run_command(
            [
                sys.executable,
                "characterize.py",
                "-i",
                str(image_path),
                "--terminal",
                "-W",
                "12",
                "-H",
                "6",
            ]
        )
        assert_ok(terminal_preview, "terminal image preview")

        text_export = run_command(
            [
                sys.executable,
                "characterize.py",
                "-i",
                str(image_path),
                "-f",
                "txt",
                "-W",
                "12",
                "-H",
                "6",
            ]
        )
        assert_ok(text_export, "text export")
        if not output_text.exists():
            print(f"text export did not create {output_text}")
            raise SystemExit(1)

        if make_video(video_path):
            video_preview = run_command(
                [
                    sys.executable,
                    "characterize.py",
                    "-i",
                    str(video_path),
                    "--video",
                    "--terminal",
                    "--no-audio",
                    "-W",
                    "12",
                    "-H",
                    "6",
                    "-r",
                    "4",
                ],
                timeout=20,
            )
            assert_ok(video_preview, "terminal video preview")
    finally:
        if output_text.exists():
            output_text.unlink()
        if temp_path.exists():
            shutil.rmtree(temp_path)

    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
