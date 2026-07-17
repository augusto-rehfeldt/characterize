import rank
import characters

import os
import sys
import time
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
import re

from PIL import Image, ImageFont, ImageEnhance, UnidentifiedImageError

from charlib.image_utils import (
    create_char_image_dict,
    get_chars,
    unite_image,
    divide_image,
    save_image,
    save_text,
)
from charlib.terminal import render_terminal_image
from charlib.player import VideoPlayer
from charlib.file_utils import optimize_files


characterize_path = os.path.realpath(os.path.dirname(__file__))


def to_hours_minutes_seconds(seconds: float):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)


def process_routine(
    image,
    character_list,
    char_images,
    character_detail_level,
    divide_image_flag,
    output_format,
    resize,
    color,
    folder_name,
    tkinter,
):
    t_image = time.time()
    image_name = image if isinstance(image, str) else getattr(image, "filename", "image")
    image_name = image_name.replace("\\", "/")

    # Inform the GUI that processing started
    if tkinter:
        print(f"<<{image_name}<<P>>", flush=True)

    # Convert to Image object if not already
    if not isinstance(image, Image.Image):
        original_image_path = image # Store path for error message
        try:
            image = Image.open(image).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print(f"Warning: Cannot identify image file '{original_image_path}'. Skipping.")
            return [] # Indicate failure

        # Resize the image if needed (only if open succeeded)
        if resize[0]:
            factor_resize = max(
                image.size[0] / resize[1][0], image.size[1] / resize[1][1]
            )
            image = image.resize(
                (
                    int(image.size[0] / factor_resize),
                    int(image.size[1] / factor_resize),
                ),
                resample=Image.Resampling.LANCZOS,
            )

        # If the image is too big, divide it into smaller parts
        if divide_image_flag:
            image_list = divide_image(image, 408960)
        else:
            image_list = [image]
    else:
        image_list = [image.convert("RGB")]

    base_name = "".join(image_name.split("/")[-1].split(".")[:-1])
    output_root = os.path.join(characterize_path, folder_name)
    saved = []

    # Process each divided image or the whole image
    for tile_index, im in enumerate(image_list):
        im = ImageEnhance.Color(im).enhance(2)
        characters_output = get_chars(im, character_list, char_images, output_format, color=color)

        # ponytail: divided tiles get a _tN suffix instead of being stitched back together
        tile_suffix = f"_t{tile_index}" if len(image_list) > 1 else ""
        out_base = os.path.join(output_root, base_name + tile_suffix)
        saved.append(out_base)

        # If saving as text
        if "txt" in output_format:
            filename = out_base + ".txt"
            save_text(characters_output[1], filename)
            if tkinter:
                print(f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename}>>", flush=True)

        # If saving as image
        if any(ext in output_format for ext in ["png", "jpg"]):
            final_image_np = unite_image(
                characters_output[0], im.width, im.height, character_detail_level
            )
            # Overlay is exactly the size of the upscaled original: straight alpha composite
            background = im.resize(
                (im.width * character_detail_level, im.height * character_detail_level),
                resample=Image.Resampling.BOX,
            ).convert("RGBA")
            combined = Image.alpha_composite(
                background, Image.fromarray(final_image_np, "RGBA")
            )

            # Optimize the image before saving
            if color:
                combined = combined.quantize(colors=256)
            else:
                combined = combined.convert("L")

            save_image(combined, output_format, color, out_base)

            if tkinter:
                print(
                    f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{out_base + ('.png' if 'png' in output_format else '.jpg')}>>",
                    flush=True,
                )

    return saved


def input_files(text=True):
    if text:
        print(
            """\nPlease, input all file paths; or a directory containing them. Separate multiple paths using spaces. For paths containing spaces, use double ("") or single ('') quotes for every path.\n"""
        )
    choice = input("Path/s: ")
    if not choice:
        return input_files()
    if choice.count('"') >= 2:
        paths = [x.strip() for x in re.findall(r'"([^"]*)"', choice)]
    elif choice.count("'") >= 2:
        paths = [x.strip() for x in re.findall(r"'([^']*)'", choice)]
    else:
        paths = [x.strip() for x in choice.split()]
    paths = [x for x in paths if os.path.exists(x)]
    return paths if len(paths) > 0 else input_files()


VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"
}

DEFAULT_FONTS = {
    "ascii": "arial.ttf",
    "arabic": "arial.ttf",
    "braille": "seguisym.ttf",
    "emoji": "seguiemj.ttf",
    "chinese": "msyh.ttc",
    "simple": "arial.ttf",
    "numbers+": "arial.ttf",
    "roman": "times.ttf",
    "numbers": "arial.ttf",
    "latin": "arial.ttf",
    "hiragana": "msyh.ttc",
    "katakana": "msyh.ttc",
    "kanji": "msyh.ttc",
    "cyrillic": "arial.ttf",
    "hangul": "malgunbd.ttf",
}

DETAIL_BY_LANGUAGE = {
    "braille": 16,
    "hiragana": 15,
    "katakana": 15,
    "kanji": 15,
    "chinese": 15,
    "hangul": 15,
    "arabic": 15,
}

LANGUAGES = list(DEFAULT_FONTS.keys())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Characterize images or play videos from one CLI"
    )
    parser.add_argument(
        "-i",
        "--input",
        "--i",
        nargs="*",
        default=[],
        help="Input file paths or folders",
    )
    parser.add_argument(
        "-W",
        "--width",
        "--cr",
        type=int,
        default=80,
        help="Render width in characters",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=None,
        help="Render height in characters (auto if unset)",
    )
    parser.add_argument(
        "-l",
        "--language",
        "--l",
        default="ascii",
        choices=LANGUAGES,
        help="Character set",
    )
    parser.add_argument(
        "-C",
        "--complexity",
        "--cl",
        type=int,
        default=12,
        help="Number of characters in charset",
    )
    parser.add_argument(
        "--empty",
        "--ec",
        action="store_true",
        help="Render darkest areas as blank",
    )
    parser.add_argument(
        "-r",
        "--framerate",
        type=float,
        default=None,
        help="Target frames per second (video mode)",
    )
    parser.add_argument(
        "-c",
        "--color",
        "--c",
        action="store_true",
        help="Enable color output",
    )
    parser.add_argument(
        "--true-color",
        action="store_true",
        help="Use actual pixel colors for terminal/video rendering",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Render output directly in the terminal",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Force video playback mode",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio during video playback",
    )
    parser.add_argument(
        "-f",
        "--format",
        "--f",
        default="png",
        help="Image output format(s): png, jpg, txt",
    )
    parser.add_argument(
        "-d",
        "--divide",
        "--d",
        action="store_true",
        help="Subdivide large images before processing",
    )
    parser.add_argument(
        "-o",
        "--optimize",
        "--o",
        action="store_true",
        help="Optimize rendered image outputs when possible",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan folders recursively when collecting image inputs",
    )
    parser.add_argument(
        "--tk",
        "--tkinter",
        dest="tkinter",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: emit <<path<<status>> progress markers for the GUI
    )
    return parser


def parse_arguments(argv=None):
    args = build_parser().parse_args(argv)
    if args.width <= 0:
        raise SystemExit("Error: --width must be a positive integer.")
    if args.height is not None and args.height <= 0:
        raise SystemExit("Error: --height must be a positive integer.")
    if args.complexity <= 0:
        raise SystemExit("Error: --complexity must be a positive integer.")
    if args.framerate is not None and args.framerate <= 0:
        raise SystemExit("Error: --framerate must be a positive number.")
    return args


def is_image_file(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def is_video_file(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def collect_image_inputs(paths, recursive=False):
    image_list = []
    for raw_path in paths:
        raw_path = raw_path.strip()
        if not raw_path or not os.path.exists(raw_path):
            continue
        if os.path.isdir(raw_path):
            if recursive:
                for root, _, files in os.walk(raw_path):
                    for filename in files:
                        candidate = os.path.join(root, filename)
                        if is_image_file(candidate):
                            image_list.append(candidate)
            else:
                for filename in os.listdir(raw_path):
                    candidate = os.path.join(raw_path, filename)
                    if os.path.isfile(candidate) and is_image_file(candidate):
                        image_list.append(candidate)
        elif is_image_file(raw_path):
            image_list.append(raw_path)
    return image_list


def collect_video_input(paths, recursive=False):
    candidates = []
    for raw_path in paths:
        raw_path = raw_path.strip()
        if not raw_path or not os.path.exists(raw_path):
            continue
        if os.path.isdir(raw_path):
            if recursive:
                walker = os.walk(raw_path)
            else:
                walker = [(raw_path, [], os.listdir(raw_path))]
            for root, _, files in walker:
                for filename in files:
                    candidate = os.path.join(root, filename)
                    if os.path.isfile(candidate) and is_video_file(candidate):
                        candidates.append(candidate)
        elif is_video_file(raw_path):
            candidates.append(raw_path)

    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"Info: multiple video candidates found. Using '{candidates[0]}'.")
    return candidates[0]


def resolve_font(language):
    font_file = DEFAULT_FONTS[language]
    font_paths = [
        font_file,
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft/Windows/Fonts",
            font_file,
        ),
    ]

    for font_path in font_paths:
        try:
            ImageFont.truetype(font_path, 10)
            return font_path
        except OSError:
            continue
    raise FileNotFoundError(f"Font {font_file} for '{language}' not found")


def build_character_resources(language, complexity_level, empty_char):
    detail_level = DETAIL_BY_LANGUAGE.get(language, 12)
    font_file = resolve_font(language)

    start_time = time.time()
    font_base = os.path.splitext(os.path.basename(DEFAULT_FONTS[language]))[0]
    cache_filename = (
        f"chars_{language}-{detail_level}-{complexity_level}-{font_base}"
        f"{'-empty' if empty_char else ''}.list"
    )
    cache_path = os.path.join(characterize_path, "dict_characters", cache_filename)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_file:
            char_list_original = pickle.load(cache_file)
        print(f"\nCharacter list loaded in {to_hours_minutes_seconds(time.time() - start_time)}")
    else:
        print("\nCreating character ranking by brightness levels...")
        char_list_original = rank.create_ranking(
            detail_level,
            font=font_file,
            list_size=complexity_level,
            allowed_characters=characters.dict_caracteres[language],
        )
        if empty_char:
            char_list_original.append((" ", 0))
            char_list_original.sort(key=lambda x: x[1])
            char_list_original = char_list_original[:complexity_level]
        with open(cache_path, "wb") as cache_file:
            pickle.dump(char_list_original, cache_file)
        print(
            f"Character list created in {round(time.time() - start_time, 2)} seconds\n"
        )

    char_list = [char for char, _ in char_list_original]
    return char_list, char_list_original, font_file, detail_level


def run_terminal_preview(image_list, char_list, width, height, color, true_color):
    for index, image_path in enumerate(image_list):
        with Image.open(image_path) as image:
            pil_image = image.convert("RGB")
            preview_height = height or max(
                1,
                int(pil_image.height / pil_image.width * width * 0.5),
            )
            header = f"{index + 1}/{len(image_list)}  {os.path.basename(image_path)}"
            render_terminal_image(
                pil_image,
                char_list,
                width,
                preview_height,
                color=color,
                true_color=true_color,
                clear=True,
                header=header,
            )


def run_image_mode(args, image_list):
    char_list, char_list_original, font_file, detail_level = build_character_resources(
        args.language,
        args.complexity,
        args.empty,
    )

    if len(char_list) <= 30:
        print("Characters to use:", char_list_original, "\n")

    if args.terminal:
        run_terminal_preview(
            image_list,
            char_list,
            args.width,
            args.height,
            args.color,
            args.true_color,
        )
        return

    folder_name = f"output/{args.language}"
    output_root = os.path.join(characterize_path, folder_name)
    os.makedirs(output_root, exist_ok=True)

    requested_formats = [
        token
        for token in re.split(r"[,\s]+", args.format.strip())
        if token
    ]

    output_format = [
        token for token in requested_formats if token in {"png", "jpg", "txt"}
    ]
    if not output_format:
        output_format = ["png"]

    char_images = None
    if any(ext in output_format for ext in ["jpg", "png"]):
        t4 = time.time()
        char_images = create_char_image_dict(
            char_list,
            detail_level,
            font_file,
            args.color,
        )
        print(f"Characters dict created in {round(time.time() - t4, 2)} seconds.\n")

    t = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() >= 2 else 1
    num_iterations = len(image_list)
    print(
        f"Processing {num_iterations} {'image' if num_iterations == 1 else 'images'}"
        f"{' in ' + str(math.ceil(num_iterations / t)) + ' cycles' if not t > num_iterations else ''}...",
        end="\n\n",
    )

    t0 = time.time()
    resize = (True, (args.width, args.height or args.width))
    divide_image_flag = args.divide

    if num_iterations == 1:
        internal_time = time.time()
        results = process_routine(
            image_list[0],
            char_list,
            char_images,
            detail_level,
            divide_image_flag,
            output_format,
            resize,
            args.color,
            folder_name,
            args.tkinter,
        )
        print(f"Elapsed time: {to_hours_minutes_seconds(time.time() - internal_time)}")
    else:
        results = []
        with ProcessPoolExecutor(max_workers=t) as executor:
            futures = {
                executor.submit(
                    process_routine,
                    image_list[i],
                    char_list,
                    char_images,
                    detail_level,
                    divide_image_flag,
                    output_format,
                    resize,
                    args.color,
                    folder_name,
                    args.tkinter,
                ): i
                for i in range(num_iterations)
            }
            start_cycle = time.time()
            batch_count = 0
            processed_count = 0
            for future in as_completed(futures):
                processed_count += 1
                batch_count += 1
                results.extend(future.result())
                # ponytail: progress prints are suppressed in tkinter mode to keep the marker protocol clean
                if not args.tkinter and (batch_count == t or processed_count == num_iterations):
                    cycle_time = time.time() - start_cycle
                    avg_time = cycle_time / batch_count
                    remaining = num_iterations - processed_count
                    estimated_remaining = remaining * avg_time
                    print(
                        f"\rProcessed {processed_count}/{num_iterations} images. "
                        f"Batch cycle time: {round(cycle_time, 2)} sec. "
                        f"Estimated remaining: {to_hours_minutes_seconds(estimated_remaining)}",
                        end="",
                        flush=True,
                    )
                    batch_count = 0
                    start_cycle = time.time()
            print(f"Total execution time: {to_hours_minutes_seconds(time.time() - t0)}")

    results = [f"{r}.{f}" for r in results for f in output_format]

    if args.optimize and len(results) <= 300:
        if os.path.exists("C:/Program Files/FileOptimizer/FileOptimizer64.exe"):
            print("\n\nOptimizing files in the background...")
            optimize_files(results, t)
            print("File optimization completed.")
        else:
            print("\nFileOptimizer not found. Skipping optimization...")
    else:
        print("\nBypassing file optimization...")

    print(
        "\n"
        + f"All done. Characterized images can be found in {output_root}."
    )


def run_video_mode(args, input_path):
    video_args = argparse.Namespace(
        input=input_path,
        width=args.width,
        height=args.height,
        language=args.language,
        complexity=args.complexity,
        empty=args.empty,
        framerate=args.framerate,
        color=args.color,
        terminal=args.terminal,
        true_color=args.true_color,
        no_audio=args.no_audio,
    )
    player = VideoPlayer(video_args)
    player.run()


def main(argv=None):
    if not os.path.exists(os.path.join(characterize_path, "output")):
        os.makedirs(os.path.join(characterize_path, "output"))
    if not os.path.exists(os.path.join(characterize_path, "dict_characters")):
        os.makedirs(os.path.join(characterize_path, "dict_characters"))

    args = parse_arguments(argv)
    if not args.input:
        args.input = input_files()

    if not args.input:
        print("No inputs provided. Closing...")
        sys.exit(1)

    video_input = collect_video_input(args.input, recursive=args.recursive)
    if args.video or video_input:
        if video_input:
            run_video_mode(args, video_input)
            return
        image_list = collect_image_inputs(args.input, recursive=args.recursive)
        if image_list:
            print("Info: no video candidates found; previewing images in terminal mode.")
            args.terminal = True
            run_image_mode(args, image_list)
            return
        print("No video input found. Closing...")
        sys.exit(1)

    image_list = collect_image_inputs(args.input, recursive=args.recursive)
    if not image_list:
        print("No images provided. Closing...")
        sys.exit(1)

    run_image_mode(args, image_list)


if __name__ == "__main__":
    main()
