To run the command line version, for now use "python [path to characterize.py] (arguments)"

Arguments:
  - (-i, --i) -> input file paths MANY ["path1", "path2", ...]
  - (-cr, --cr') -> character resolution parameter ONE (1 to 4000) [THe image will be automatically divided around the 800-1000                      characters resolution mark]
  - (-cl, --cl) -> complexity level parameter ONE (1 to 40)
  - (-l, --l) -> language parameter ONE [ascii, chinese, ...] [Available languages: "ascii", "arabic", "braille", "emoji", "chinese", "simple", "numbers+", "roman", "numbers", "latin", "hiragana", "katakana", "kanji", "cyrillic", "hangul"]
  - (-d, --d) -> divide parameter ONE (true/false)
  - (-c, --c) -> color parameter ONE (true/false)
  - (-f, --f) -> format parameter MANY [png, jpg, txt]
  - (-o, --o) -> optimize parameter ONE (true/false) [for now, requires having FileOptimizer64.exe on C:/Program^                                                                       Files/FileOptimizer]

Pass them like "python [path to characterize.py] -i PATH -c false -o true" and so on. If you don't pass arguments, the program will ask you for the parameters using user inputs.
