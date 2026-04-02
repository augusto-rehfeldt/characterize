#!/usr/bin/env python3
"""Compatibility wrapper for the unified characterize.py entrypoint."""

import sys

from characterize import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--video" not in argv:
        argv = ["--video", *argv]
    main(argv)
