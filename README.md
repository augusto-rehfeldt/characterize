To run the command line image tooling, use:

```bash
python characterize.py [arguments]
```

To run the video / terminal playback tool, use:

```bash
python characterize_video.py [arguments]
```

`characterize_video.py` supports a `--terminal` flag that prints the output directly in the terminal instead of opening a window.
It also automatically switches to terminal mode for still-image inputs.

Arguments:
  - (-i, --i) -> input file paths MANY ["path1", "path2", ...]
  - (-cr, --cr') -> character resolution parameter ONE (1 to 4000) [THe image will be automatically divided around the 800-1000                      characters resolution mark]
  - (-cl, --cl) -> complexity level parameter ONE (1 to 40)
  - (-l, --l) -> language parameter ONE [ascii, chinese, ...] [Available languages: "ascii", "arabic", "braille", "emoji", "chinese", "simple", "numbers+", "roman", "numbers", "latin", "hiragana", "katakana", "kanji", "cyrillic", "hangul"]
  - (-d, --d) -> divide parameter ONE (true/false)
  - (-c, --c) -> color parameter ONE (true/false)
  - (-f, --f) -> format parameter MANY [png, jpg, txt]
  - (-ec, --ec) -> empty character parameter ONE (true/false)
  - (-o, --o) -> optimize parameter ONE (true/false) [for now, requires having FileOptimizer64.exe on C:/Program Files/FileOptimizer/]

Pass them like "python [path to characterize.py] -i PATH -c false -o true" and so on. If you don't pass arguments, the program will ask you for the parameters using user inputs.

Video playback examples:

```bash
python characterize_video.py -i path/to/video.mp4 --terminal -W 80 --color
python characterize_video.py -i path/to/image.png --terminal -W 80
```

---

Output:

