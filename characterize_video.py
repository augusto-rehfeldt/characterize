#!/usr/bin/env python3
import argparse
import sys
import os
from charlib.player import VideoPlayer

def create_parser():
    p = argparse.ArgumentParser(description="Real-time ASCII video playback")
    p.add_argument("-i", "--input", required=False,
                   help="Path to video file")
    p.add_argument("-W", "--width", type=int, default=80,
                   help="Output width in characters")
    p.add_argument("-H", "--height", type=int,
                   help="Output height in characters (auto if unset)")
    p.add_argument("-l", "--language", default="ascii",
                   choices=['ascii', 'arabic', 'braille', 'emoji', 'chinese', 'simple', 'numbers+', 'roman', 'numbers', 'latin', 'hiragana', 'katakana', 'kanji', 'cyrillic', 'hangul'],
                   help="Character set")
    p.add_argument("-c", "--complexity", type=int, default=12,
                   help="Number of characters in charset")
    p.add_argument("--empty", action="store_true",
                   help="Render darkest areas as blank")
    p.add_argument("-r", "--framerate", type=float, default=None,
                   help="Target frames per second (default: video's FPS or 24.0)")
    p.add_argument("--color", action="store_true",
                   help="Enable color output (terminal and windowed)")
    p.add_argument("--terminal", action="store_true",
                   help="Output to terminal instead of a window")
    p.add_argument("--true-color", action="store_true",
                   help="Use actual pixel colors instead of brightness-based colors")
    return p

def main():
    parser = create_parser()
    
    if len(sys.argv) == 1:
        print("No command-line arguments provided. Entering interactive mode.")
        video_input_path = input("Enter video file path: ").strip('\'"')
        if not video_input_path or not os.path.isfile(video_input_path):
            print(f"Error: Video file '{video_input_path}' not found or invalid.")
            sys.exit(1)
        width_str = input(f"Output width in characters (default: 80, press Enter): ")
        parsed_cli_args = ['-i', video_input_path]
        if width_str:
            try:
                int(width_str)
                parsed_cli_args.extend(['-W', width_str])
            except ValueError:
                print(f"Warning: Invalid width '{width_str}', using default 80.")
        args = parser.parse_args(parsed_cli_args)
    else:
        args = parser.parse_args()
        if not args.input or not os.path.isfile(args.input):
            print(f"Error: video file '{args.input}' not found. Please provide a valid path.")
            sys.exit(1)

    player = VideoPlayer(args)
    player.run()

if __name__ == "__main__":
    main()