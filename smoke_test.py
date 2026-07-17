import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
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
    output_png = ROOT / "output" / "ascii" / f"{unique}.png"

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
                "main.py",
                "charlib/player.py",
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

        png_export = run_command(
            [
                sys.executable,
                "characterize.py",
                "-i",
                str(image_path),
                "-f",
                "png",
                "-W",
                "12",
                "-H",
                "6",
            ]
        )
        assert_ok(png_export, "png export")
        if not output_png.exists():
            print(f"png export did not create {output_png}")
            raise SystemExit(1)

        # GUI integration: --tk must be accepted and emit <<path<<status>> markers
        tk_protocol = run_command(
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
                "--tk",
            ]
        )
        assert_ok(tk_protocol, "tk protocol run")
        expected_marker = f"<<{str(image_path).replace(os.sep, '/')}<<P>>"
        if expected_marker not in tk_protocol.stdout.replace("\\", "/"):
            print("tk protocol markers missing from stdout")
            print(tk_protocol.stdout)
            raise SystemExit(1)

        browser = subprocess.Popen(
            [sys.executable, "browser_mode.py"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            first_line = browser.stdout.readline()
            match = re.search(r"http://\S+", first_line)
            if not match:
                print(f"browser mode did not print a URL, got: {first_line!r}")
                raise SystemExit(1)
            base_url = match.group(0)

            for path, marker in (
                ("/", b"<canvas"),
                ("/app.js", b"CHARSETS"),
                ("/style.css", b".card"),
            ):
                with urllib.request.urlopen(base_url + path, timeout=10) as response:
                    if response.status != 200 or marker not in response.read():
                        print(f"browser mode static check failed for {path}")
                        raise SystemExit(1)
        finally:
            browser.terminate()

        if shutil.which("node"):
            node_check = run_command(
                [
                    "node",
                    "-e",
                    "const m = require('./docs/app.js');"
                    "const g = m.grayFromRgba([255,255,255,255, 0,0,0,255]);"
                    "if (g[0] !== 1 || g[1] !== 0) throw new Error('grayFromRgba');"
                    "if (m.percentile([0, 0.5, 1], 90) !== 0.9) throw new Error('percentile');"
                    "const a = m.amplifyGray(new Float64Array([0.5]), 0.5);"
                    "if (a[0] !== 0.5) throw new Error('amplifyGray');"
                    "if (m.mapIndex(0.99, 12) !== 11 || m.mapIndex(0, 12) !== 0) throw new Error('mapIndex');"
                    "const cs = m.buildCharset([['a',0],['b',10],['c',20],['d',30]], 2, false);"
                    "if (cs.length !== 2 || cs[0] !== 'a') throw new Error('buildCharset');"
                    "const cse = m.buildCharset([['a',0],['b',10],['c',20],['d',30]], 2, true);"
                    "if (cse[0] !== ' ') throw new Error('buildCharset empty');",
                ]
            )
            assert_ok(node_check, "browser app logic check (node)")

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
        for artifact in (output_text, output_png):
            if artifact.exists():
                artifact.unlink()
        if temp_path.exists():
            shutil.rmtree(temp_path)

    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
