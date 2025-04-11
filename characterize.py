import rank
import characters

import os
import sys
import time
import math
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import argparse
import re

import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, UnidentifiedImageError


characterize_path = os.path.realpath(os.path.dirname(__file__))


def amplify_differences(values, threshold):
    values = np.array(values)
    deviations = np.abs(values - threshold)

    amplification_factors = np.where(
        values >= threshold, 1 + deviations, 1 - deviations
    )
    amplified_values = np.clip(values * amplification_factors, 0, 1)

    return amplified_values


def create_char_image(char, color, detail, font):
    # create a new image
    new_image = Image.new("RGBA", (detail, detail), color=(0, 0, 0, 255))
    # get the font
    font = ImageFont.truetype(font, detail)
    # draw the text
    draw = ImageDraw.Draw(new_image)
    draw.fontmode = "L"
    draw.text(
        ((detail) / 2, (detail) / 2),
        char,
        align="center",
        font=font,
        fill=color,
        anchor="mm",
    )
    # return the new image
    return new_image


def create_char_image_dict(characters, detail, font, color=False):
    char_images = {}
    for char in characters:
        char_images[char] = create_char_image(
            char, (255, 255, 255, 255) if not color else (0, 0, 0, 0), detail, font
        )
    return char_images


def to_hours_minutes_seconds(seconds: float):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)


def get_chars(image, char_list, char_images, format, color):
    # Convert image to grayscale array normalized between 0 and 1
    gray_arr = np.array(image.convert("L")) / 255.0

    if not color:
        # Compute threshold and amplify differences vectorially
        threshold = np.percentile(gray_arr, 90)
        amplified = amplify_differences(gray_arr, threshold)
        indices = (amplified * len(char_list) - 0.5).astype(int)
    else:
        indices = (gray_arr * len(char_list) - 0.5).astype(int)

    # Ensure indices are within valid bounds
    indices = np.clip(indices, 0, len(char_list) - 1)

    # Build list for image output format using vectorized operations
    if any(x in format for x in ["png", "jpg"]):
        characters_output = [
            [char_images[char_list[ix]] for ix in row]
            for row in indices.tolist()
        ]
    else:
        characters_output = []

    if any(x in format for x in ["txt"]):
        # Use NumPy indexing for faster mapping
        char_array = np.array(char_list) # Convert list of characters to NumPy array
        mapped_chars = char_array[indices] # Map indices to characters using fancy indexing
        # Mimic original double flip and transpose logic for text output
        characters_aux = np.fliplr(np.fliplr(mapped_chars.transpose())).tolist()
    else:
        characters_aux = []

    return characters_output, characters_aux


def unite_image(characters, original_width, original_height, detail_level):
    """
    Assembles the final image from individual character images using NumPy for speed.
    """
    # Create an empty NumPy array for the final image
    final_height = original_height * detail_level
    final_width = original_width * detail_level
    # Assuming RGBA characters based on previous logic
    final_image_np = np.zeros((final_height, final_width, 4), dtype=np.uint8)

    # characters is organized as rows (height) then columns (width)
    for j in range(original_height):
        for i in range(original_width):
            # Convert character PIL image to NumPy array
            char_image_np = np.array(characters[j][i].convert("RGBA"))
            
            # Calculate the slice coordinates
            y_start = j * detail_level
            y_end = y_start + detail_level
            x_start = i * detail_level
            x_end = x_start + detail_level
            
            # Place the character array into the final image array
            # Ensure the character image has the expected dimensions
            if char_image_np.shape == (detail_level, detail_level, 4):
                 final_image_np[y_start:y_end, x_start:x_end] = char_image_np
            else:
                 # Handle potential size mismatches if necessary (e.g., resize or log warning)
                 # For now, we assume correct size based on create_char_image
                 print(f"Warning: Character image size mismatch at ({i},{j}). Expected ({detail_level},{detail_level}), got {char_image_np.shape[:2]}")
                 # Attempt to paste anyway if dimensions allow, might need resizing logic
                 try:
                     final_image_np[y_start:y_end, x_start:x_end] = char_image_np[:detail_level, :detail_level]
                 except ValueError:
                     print(f"Error: Could not place character at ({i},{j}) due to size mismatch.")


    # Convert the final NumPy array back to a PIL Image
    new_image = Image.fromarray(final_image_np, 'RGBA')
    return new_image


def divide_image(image, min_size):
    """
    Divide the image into smaller parts if its size exceeds the min_size.
    """
    image_list = [image]

    while image_list[0].size[0] * image_list[0].size[1] >= min_size:
        temp_list = []
        for img in image_list:
            temp_list.extend(
                [
                    img.crop((0, 0, img.width // 2, img.height // 2)),
                    img.crop((img.width // 2, 0, img.width, img.height // 2)),
                    img.crop((0, img.height // 2, img.width // 2, img.height)),
                    img.crop((img.width // 2, img.height // 2, img.width, img.height)),
                ]
            )
        image_list = temp_list.copy()

    return image_list


def save_image(image, format, color, filename, max_attempts=10):
    save_options = {
        "png": {"format": "PNG", "compress_level": 9},
        "jpg": {"format": "JPEG", "quality": 95},
    }

    for fmt in ["png", "jpg"]:
        if fmt in format:
            attempt = 0
            while attempt < max_attempts:
                try:
                    # Ensure the image is in RGB mode
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Apply color quantization if needed
                    if color and fmt == "png":
                        image = image.quantize(colors=256)

                    # Save the image
                    image.save(f"{filename}.{fmt}", **save_options[fmt])

                    # Verify the saved image
                    with Image.open(f"{filename}.{fmt}") as img:
                        img.verify()

                    break  # Exit the loop if successful
                except Exception as e:
                    attempt += 1
                    if attempt < max_attempts:
                        time.sleep(1)  # Wait a bit before retrying
                    else:
                        # If all attempts fail, try to save as a different format
                        try:
                            backup_format = "jpg" if fmt == "png" else "png"
                            image.save(
                                f"{filename}_backup.{backup_format}",
                                **save_options[backup_format],
                            )
                            print(f"Saved backup as {filename}_backup.{backup_format}")
                        except Exception as backup_e:
                            print(f"Failed to save backup: {str(backup_e)}")


def save_text(characters, filename):
    with open(filename, "w") as f:
        f.writelines([" ".join(line) + "\n" for line in characters])


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
    image_name = "".join([x for x in image])

    # Inform if tkinter is being used
    if tkinter:
        print(f"<<{image_name}<<P>>")

    # Convert to Image object if not already
    if not isinstance(image, Image.Image):
        original_image_path = image # Store path for error message
        try:
            image = Image.open(image).convert("RGB")
        except UnidentifiedImageError:
            print(f"Warning: Cannot identify image file '{original_image_path}'. Skipping.")
            return None # Indicate failure

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
        image = image.convert("RGB")
        image_list = [image]

    # Process each divided image or the whole image
    for im in image_list:
        im = ImageEnhance.Color(im).enhance(2)
        characters_output = get_chars(im, character_list, char_images, output_format, color=color)

        # If saving as text
        if "txt" in output_format:
            image_name = image_name.replace("\\", "/")
            filename = os.path.join(
                os.path.join(characterize_path, folder_name),
                "".join(image_name.split("/")[-1].split(".")[:-1]) + ".txt",
            )
            save_text(characters_output[1], filename)
            if tkinter:
                print(f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename}>>")

        # If saving as image
        if any(ext in output_format for ext in ["png", "jpg"]):
            final_image = unite_image(
                characters_output[0], im.width, im.height, character_detail_level
            )
            # Resize original image to match final output size
            im_resized = im.resize(
                (im.width * character_detail_level, im.height * character_detail_level),
                resample=Image.Resampling.BOX
            )

            # Convert images to NumPy arrays (RGBA)
            im_np = np.array(im_resized.convert("RGBA"))
            final_image_np = np.array(final_image.convert("RGBA"))

            # Calculate offset for centering final_image on im_resized
            bg_h, bg_w, _ = im_np.shape
            img_h, img_w, _ = final_image_np.shape
            y_offset = (bg_h - img_h) // 2
            x_offset = (bg_w - img_w) // 2

            # Create combined image array (copy of background)
            combined_np = im_np.copy()

            # Get the alpha channel of the overlay
            alpha = final_image_np[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis] # Reshape for broadcasting

            # Alpha compositing: combined = overlay * alpha + background * (1 - alpha)
            # Ensure slicing bounds are valid
            y_start, y_end = max(0, y_offset), min(bg_h, y_offset + img_h)
            x_start, x_end = max(0, x_offset), min(bg_w, x_offset + img_w)
            
            overlay_y_start = max(0, -y_offset)
            overlay_x_start = max(0, -x_offset)
            overlay_y_end = overlay_y_start + (y_end - y_start)
            overlay_x_end = overlay_x_start + (x_end - x_start)

            if y_end > y_start and x_end > x_start: # Check if there is overlap
                background_slice = combined_np[y_start:y_end, x_start:x_end]
                overlay_slice = final_image_np[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
                alpha_slice = alpha[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]

                # Perform blending only if shapes match
                if background_slice.shape == overlay_slice.shape:
                    combined_slice = overlay_slice * alpha_slice + background_slice * (1 - alpha_slice)
                    combined_np[y_start:y_end, x_start:x_end] = combined_slice.astype(np.uint8)
                else:
                     print(f"Warning: Shape mismatch during alpha compositing. BG: {background_slice.shape}, Overlay: {overlay_slice.shape}")
                     # Fallback or alternative handling if needed

            # Convert back to PIL Image
            combined = Image.fromarray(combined_np, 'RGBA')

            # Convert back to RGB for saving
            combined = combined.convert("RGB")

            # Optimize the image before saving
            if color:
                combined = combined.quantize(colors=256)
            else:
                combined = combined.convert("L")

            # Save the image
            image_name = image_name.replace("\\", "/")
            filename = os.path.join(
                os.path.join(characterize_path, folder_name),
                "".join(image_name.split("/")[-1].split(".")[:-1]),
            )

            # Start the thread for saving the image
            save_image(combined, output_format, color, filename)

            if tkinter:
                print(
                    f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename+'.png' if 'png' in output_format else filename+'.jpg'}>>"
                )

            return filename


def choose_option(what, options_list, text=True):
    if text:
        list_as_items = [
            str(i + 1) + ") " + str(item) + "\n" for i, item in enumerate(options_list)
        ]
        items = ""
        for item in list_as_items:
            items += item
        print(f"\nWhich of the following {what} do you want to use?\n\n{items}")
    try:
        choice = int(input("Choice: "))
    except ValueError:
        return choose_option(what, options_list, False)
    else:
        if not 0 < choice < len(options_list) + 1:
            return choose_option(what, options_list, False)
        return options_list[choice - 1]


def choose_value(what, min=False, max=False, text=True):
    if text:
        print(
            f"\nPlease enter a {what} integer value {'between '+str(min)+' and '+str(max)+' ' if min and max else ('above '+str(min) if min else 'below '+str(max))}(float values will be converted to integers by rounding them down).\n"
        )
    try:
        choice = int(input("Choice: "))
    except ValueError:
        return choose_value(what, min, max, False)
    else:
        if max and choice > max:
            print(f"Choose a value below {max+1}.")
            return choose_value(what, min, max, False)
        elif min and choice < min:
            print(f"Choose a value above {max+1}.")
            return choose_value(what, min, max, False)
        return choice


def binary_choice(what, text=True):
    if text:
        print(f"\nDo you want to {what}?\n\n1) Yes\n2) No\n")
    try:
        choice = int(input("Choice: "))
    except ValueError:
        return binary_choice(what, False)
    else:
        if not choice in [1, 2]:
            return binary_choice(what, False)
        return True if choice == 1 else False


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


def parse_arguments():
    # Simplify boolean argument parsing
    def str_to_bool(value):
        return value.lower() in ['true', 't', 'yes', 'y']

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--i", type=str, nargs="*", help='input file paths MANY ["path1", "path2", ...]')
    parser.add_argument("-cr", "--cr", type=int, help="character resolution parameter ONE (1 to 4000)")
    parser.add_argument("-cl", "--cl", type=int, help="complexity level parameter ONE (1 to 4000)")
    parser.add_argument("-l", "--l", type=str, help="language parameter ONE [ascii, chinese, ...]")
    parser.add_argument("-d", "--d", type=str, help="divide parameter ONE (true/false)")
    parser.add_argument("-c", "--c", type=str, help="color parameter ONE (true/false)")
    parser.add_argument("-f", "--f", nargs="+", help="format parameter MANY [png, jpg, txt]")
    parser.add_argument("-o", "--o", type=str, help="optimize parameter ONE (true/false)")
    parser.add_argument("-ec", "--ec", type=str, help="empty character parameter ONE (true/false)")
    parser.add_argument("-tk", "--tk", type=lambda x: str_to_bool(x), help="tkinter parameter ONE (true/false)")
    args = parser.parse_args()

    input_files = args.i
    cr = (True, (args.cr, args.cr)) if isinstance(args.cr, int) else (False, False)
    cl = args.cl
    l = args.l

    ec = (True, str_to_bool(args.ec)) if args.ec else (False, False)
    d = (True, str_to_bool(args.d)) if args.d else (False, False)
    c = (True, str_to_bool(args.c)) if args.c else (False, False)
    f = " ".join(args.f) if args.f else None
    o = (True, str_to_bool(args.o)) if args.o else (False, False)

    tk = args.tk

    return l, cl, c, input_files, cr, ec, d, f, o, tk


def divide_list(lst, n):
    """Divide a list into n approximately equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def optimize_file(file_path):
    cmd = f'"C:/Program Files/FileOptimizer/FileOptimizer64.exe" "{file_path}"'
    subprocess.run(cmd, shell=True, check=True)


def optimize_files(files, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(optimize_file, files)


def process_image(image, character_list, char_images, character_detail_level, divide_image_flag, output_format, resize, color, folder_name, tkinter):
    return process_routine(
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
    )


if __name__ == "__main__":
    if not os.path.exists(os.path.join(characterize_path, "output")):
        os.makedirs(os.path.join(characterize_path, "output"))
    if not os.path.exists(os.path.join(characterize_path, "dict_characters")):
        os.makedirs(os.path.join(characterize_path, "dict_characters"))

    fonts = {
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

    languages = list(fonts.keys())

    (
        language,
        complexity_level,
        color,
        image_src,
        resize,
        empty_char,
        divide_image_flag,
        output_format,
        optimize,
        tkinter,
    ) = parse_arguments()

    if not language or not language in languages:
        language = choose_option("characters", sorted(languages))
    if not complexity_level or not complexity_level in range(1, 41):
        complexity_level = choose_value("complexity level", 1, 40)
    if len(color) != 2 or not color[0]:
        color = binary_choice("use color")
    else:
        color = color[1]
    if not image_src:
        image_src = input_files()
    if not isinstance(resize, tuple) or not resize[0]:
        choice = choose_value("character resolution", 1, 4000)
        resize = (True, (choice, choice))
    if not divide_image_flag[0]:
        divide_image_flag = binary_choice("subdivide the image")
    else:
        divide_image_flag = divide_image_flag[1]
    if len(empty_char) != 2 or not empty_char[0]:
        empty_char = binary_choice("use an empty character to represent the darkest pixels")
    else:
        empty_char = empty_char[1]
    if not output_format or not any(
        x in ["png", "jpg", "txt"] for x in output_format.split()
    ):
        output_format = choose_option(
            "file formats",
            ["png", "jpg", "txt", "txt, png", "txt, jpg", "png, jpg", "png, jpg, txt"],
        )
    if not optimize[0]:
        optimize = binary_choice("optimize the resulting images (when images <= 300)")
    else:
        optimize = optimize[1]

    folder_name = f"output/{language}"

    if not os.path.exists(os.path.join(characterize_path, folder_name)):
        os.makedirs(os.path.join(characterize_path, folder_name))

    if "txt" in output_format:
        if not os.path.exists(os.path.join(characterize_path, f"output/{language}/text")):
            os.makedirs(os.path.join(characterize_path, f"output/{language}/text"))

    if not any(x in output_format for x in ("png", "jpg", "txt")):
        output_format = "png"

    image_list = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.jfif', '.webp'}

    for path in image_src:
        path = path.strip()
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    image_list.append(os.path.join(path, filename))
        else:
            if any(path.lower().endswith(ext) for ext in valid_extensions):
                image_list.append(path)

    if not image_list:
        print("No images provided. Closing...")
        sys.exit()

    # Determine character detail level based on language
    detail_level = {
        "braille": 16,
        "hiragana": 15, "katakana": 15, "kanji": 15,
        "chinese": 15, "hangul": 15, "arabic": 15
    }.get(language, 12)

    character_dict = characters.dict_caracteres

    # Font paths to try
    font_paths = [
        fonts[language],
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                    'Microsoft/Windows/Fonts', fonts[language])
    ]

    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 10)
            font_file = font_path
            break
        except OSError:
            continue

    if not font:
        print(f"Error: Font {fonts[language]} for '{language}' not found")
        sys.exit(1)

    start_time = time.time()

    # Generate cache filename
    font_base = os.path.splitext(os.path.basename(fonts[language]))[0]
    cache_filename = f"chars_{language}-{detail_level}-{complexity_level}-{font_base}{'-empty' if empty_char else ''}.list"
    cache_path = os.path.join(characterize_path, "dict_characters", cache_filename)

    # Try to load from cache first
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            char_list_original = pickle.load(f)
        char_list = [x[0] for x in char_list_original]
        print(f"\nCharacter list loaded in {to_hours_minutes_seconds(time.time() - start_time)}")
    else:
        print("\nCreating character ranking by brightness levels...")
        char_list_original = rank.create_ranking(
            detail_level,
            font=font_file,
            list_size=complexity_level,
            allowed_characters=character_dict[language]
        )
        
        if empty_char:
            char_list_original.append((" ", 0))
            char_list_original.sort(key=lambda x: x[1])
            char_list_original = char_list_original[:complexity_level]

        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(char_list_original, f)
        
        char_list = [x[0] for x in char_list_original]
        print(f"Character list created in {round(time.time() - start_time, 2)} seconds\n")
    t4 = time.time()
    if any(x in output_format for x in ["jpg", "png"]):
        char_images = create_char_image_dict(
            char_list, detail_level, font_file, color
        )
        print(f"Characters dict created in {round(time.time()-t4, 2)} seconds.\n")
    else:
        char_images = False
    t = os.cpu_count() // 2 if os.cpu_count() >= 2 else 1
    num_iterations = len(image_list)
    if len(char_list) <= 30:
        print("Characters to use:", char_list_original, "\n")
    print(
        f"Processing {num_iterations} {'image' if num_iterations == 1 else 'images'}{' in '+str(math.ceil(num_iterations/t))+' cycles' if not t > num_iterations else ''}...",
        end="\n\n",
    )

    t0 = time.time()

    if num_iterations == 1:
        internal_time = time.time()
        results = []
        for i, image in enumerate(image_list):
            results.append(
                process_routine(
                    image,
                    char_list,
                    char_images,
                    detail_level,
                    divide_image_flag,
                    output_format,
                    resize,
                    color,
                    folder_name,
                    tkinter,
                )
            )
        if not tkinter:
            print(f"Elapsed time: {to_hours_minutes_seconds(time.time()-internal_time)}")
    else:
        results = []
        with ProcessPoolExecutor(max_workers=t) as executor:
            futures = {executor.submit(
                    process_image,
                    image_list[i],
                    char_list,
                    char_images,
                    detail_level,
                    divide_image_flag,
                    output_format,
                    resize,
                    color,
                    folder_name,
                    tkinter,
            ): i for i in range(num_iterations)}
            start_cycle = time.time()
            batch_count = 0
            processed_count = 0
            for future in as_completed(futures):
                processed_count += 1
                batch_count += 1
                results.append(future.result())
                if batch_count == t or processed_count == num_iterations:
                    cycle_time = time.time() - start_cycle
                    avg_time = cycle_time / batch_count
                    remaining = num_iterations - processed_count
                    estimated_remaining = remaining * avg_time
                    print(f"\rProcessed {processed_count}/{num_iterations} images. Batch cycle time: {round(cycle_time, 2)} sec. Estimated remaining: {to_hours_minutes_seconds(estimated_remaining)}", end="", flush=True)
                    batch_count = 0
                    start_cycle = time.time()
            if not tkinter:
                print(f"Total execution time: {to_hours_minutes_seconds(time.time() - t0)}")

    output_format = [
        x.replace(",", "").replace(".", "").replace(";", "")
        for x in output_format.split()
        if any(z in x for z in ["png", "jpg", "txt"])
    ]

    results = [
        item
        for sublist in [[r + f".{f}" for r in results] for f in output_format]
        for item in sublist
    ]

    if optimize and len(results) <= 300:
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
        + f"All done. Characterized images can be found in {os.path.join(characterize_path, folder_name)}."
    )
