import rank
import characters

import os
import sys
import time
import math
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import argparse
import re

import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageEnhance


characterize_path = os.path.realpath(os.path.dirname(__file__))


def divide_list(lst: list, n: int):
    """Divides a list into n sublists."""
    return [lst[i::n] for i in range(n)]


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
    original_width, original_height = image.size
    pixels = image.convert("L").load()
    color_levels = [
        [pixels[i, j] / 255 for j in range(original_height)]
        for i in range(original_width)
    ]
    # exacerbate or reduce differences to get a better b&w image
    if not color:
        color_levels_list = [item for sublist in color_levels for item in sublist]
        threshold = np.percentile(color_levels_list, 90)
    for i, x in enumerate(color_levels):
        color_levels[i] = list(
            map(
                lambda n: int(n * len(char_list) - 0.5),
                amplify_differences(x, threshold) if not color else x,
            )
        )
    # create a list
    characters = (
        [[char_images[char_list[y]] for y in color_int] for color_int in color_levels]
        if any(x in format for x in ["png", "jpg"])
        else []
    )
    characters_aux = (
        np.fliplr(
            np.fliplr(
                np.array(
                    [[char_list[y] for y in color_int] for color_int in color_levels]
                ).transpose()
            )
        )
        if any(x in format for x in ["txt"])
        else []
    )
    # return the list
    return characters, characters_aux


def unite_image(characters, original_width, original_height, detail_level):
    new_image = Image.new(
        "RGBA",
        (detail_level * original_width, detail_level * original_height),
        color=(0, 0, 0, 0),
    )

    for i in range(original_width):
        for j in range(original_height):
            char_image = characters[i][j].convert("RGBA")
            new_image.paste(
                char_image, (i * detail_level, j * detail_level), char_image
            )

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


def rutina(
    image,
    char_list,
    char_images,
    detail,
    divide,
    format,
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
        image = Image.open(image).convert("RGB")

        # Resize the image if needed
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
        if divide:
            image_list = divide_image(image, 408960)
        else:
            image_list = [image]
    else:
        image = image.convert("RGB")
        image_list = [image]

    # Process each divided image or the whole image
    for im in image_list:
        im = ImageEnhance.Color(im).enhance(2)
        characters_imagen = get_chars(im, char_list, char_images, format, color=color)

        # If saving as text
        if "txt" in format:
            image_name = image_name.replace("\\", "/")
            filename = os.path.join(
                os.path.join(characterize_path, folder_name),
                "".join(image_name.split("/")[-1].split(".")[:-1]) + ".txt",
            )
            save_text(characters_imagen[1], filename)
            if tkinter:
                print(f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename}>>")

        # If saving as image
        if any(ext in format for ext in ["png", "jpg"]):
            imagen_final = unite_image(
                characters_imagen[0], im.width, im.height, detail
            )
            im = im.resize(
                (im.width * detail, im.height * detail), resample=Image.Resampling.BOX
            )
            bg_w, bg_h = im.size
            img_w, img_h = imagen_final.size
            offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)

            # Convert both images to RGBA mode before pasting
            im = im.convert("RGBA")
            imagen_final = imagen_final.convert("RGBA")

            # Create a new blank image with the same size as 'im'
            combined = Image.new("RGBA", im.size, (0, 0, 0, 0))

            # Paste 'im' onto the new image
            combined.paste(im, (0, 0))

            # Paste 'imagen_final' onto the new image
            combined.paste(imagen_final, offset, imagen_final)

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
            save_image(combined, format, color, filename)

            if tkinter:
                print(
                    f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename+'.png' if 'png' in format else filename+'.jpg'}>>"
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


def process_image(image, lista_caracteres, char_images, nivel_detalle_caracter, dividir_imagen, formato_final, resize, color, folder_name, tkinter):
    return rutina(
        image,
        lista_caracteres,
        char_images,
        nivel_detalle_caracter,
        dividir_imagen,
        formato_final,
        resize,
        color,
        folder_name,
        tkinter,
    )


if __name__ == "__main__":
    if not os.path.exists(os.path.join(characterize_path, "output")):
        os.makedirs(os.path.join(characterize_path, "output"))
    if not os.path.exists(os.path.join(characterize_path, "dict_caracteres")):
        os.makedirs(os.path.join(characterize_path, "dict_caracteres"))

    fuentes = {
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

    languages = list(fuentes.keys())

    (
        idioma,
        nivel_complejidad,
        color,
        image_src,
        resize,
        empty_char,
        dividir_imagen,
        formato_final,
        optimize,
        tkinter,
    ) = parse_arguments()

    if not idioma or not idioma in languages:
        idioma = choose_option("characters", sorted(languages))
    if not nivel_complejidad or not nivel_complejidad in range(1, 41):
        nivel_complejidad = choose_value("complexity level", 1, 40)
    if len(color) != 2 or not color[0]:
        color = binary_choice("use color")
    else:
        color = color[1]
    if not image_src:
        image_src = input_files()
    if not isinstance(resize, tuple) or not resize[0]:
        choice = choose_value("character resolution", 1, 4000)
        resize = (True, (choice, choice))
    if not dividir_imagen[0]:
        dividir_imagen = binary_choice("subdivide the image")
    else:
        dividir_imagen = dividir_imagen[1]
    if len(empty_char) != 2 or not empty_char[0]:
        empty_char = binary_choice("use an empty character to represent the darkest pixels")
    else:
        empty_char = empty_char[1]
    if not formato_final or not any(
        x in ["png", "jpg", "txt"] for x in formato_final.split()
    ):
        formato_final = choose_option(
            "file formats",
            ["png", "jpg", "txt", "txt, png", "txt, jpg", "png, jpg", "png, jpg, txt"],
        )
    if not optimize[0]:
        optimize = binary_choice("optimize the resulting images (when images <= 300)")
    else:
        optimize = optimize[1]

    folder_name = f"output/{idioma}"

    if not os.path.exists(os.path.join(characterize_path, folder_name)):
        os.makedirs(os.path.join(characterize_path, folder_name))

    if "txt" in formato_final:
        if not os.path.exists(os.path.join(characterize_path, f"output/{idioma}/text")):
            os.makedirs(os.path.join(characterize_path, f"output/{idioma}/text"))

    if not any(x in formato_final for x in ("png", "jpg", "txt")):
        formato_final = "png"

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
    }.get(idioma, 12)

    char_dict = characters.dict_caracteres

    # Font paths to try
    font_paths = [
        fuentes[idioma],
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                    'Microsoft/Windows/Fonts', fuentes[idioma])
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
        print(f"Error: Font {fuentes[idioma]} for '{idioma}' not found")
        sys.exit(1)

    start_time = time.time()

    # Generate cache filename
    font_base = os.path.splitext(os.path.basename(fuentes[idioma]))[0]
    cache_filename = f"chars_{idioma}-{detail_level}-{nivel_complejidad}-{font_base}{'-empty' if empty_char else ''}.list"
    cache_path = os.path.join(characterize_path, "dict_caracteres", cache_filename)

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
            list_size=nivel_complejidad,
            allowed_characters=char_dict[idioma]
        )
        
        if empty_char:
            char_list_original.append((" ", 0))
            char_list_original.sort(key=lambda x: x[1])
            char_list_original = char_list_original[:nivel_complejidad]

        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(char_list_original, f)
        
        char_list = [x[0] for x in char_list_original]
        print(f"Character list created in {round(time.time() - start_time, 2)} seconds\n")
    t4 = time.time()
    if any(x in formato_final for x in ["jpg", "png"]):
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
        t_interno = time.time()
        results = []
        for i, image in enumerate(image_list):
            results.append(
                rutina(
                    image,
                    char_list,
                    char_images,
                    detail_level,
                    dividir_imagen,
                    formato_final,
                    resize,
                    color,
                    folder_name,
                    tkinter,
                )
            )
        if not tkinter:
            print(f"Elapsed time: {to_hours_minutes_seconds(time.time()-t_interno)}")
    else:
        results = []
        with ProcessPoolExecutor(max_workers=t) as executor:
            futures = [
                executor.submit(
                    process_image,
                    image_list[i],
                    char_list,
                    char_images,
                    detail_level,
                    dividir_imagen,
                    formato_final,
                    resize,
                    color,
                    folder_name,
                    tkinter,
                )
                for i in range(num_iterations)
            ]
            for future in futures:
                results.append(future.result())
            if not tkinter:
                print(
                    f"Total execution time: {to_hours_minutes_seconds(round((time.time() - t0), 2))}"
                )

    formato_final = [
        x.replace(",", "").replace(".", "").replace(";", "")
        for x in formato_final.split()
        if any(z in x for z in ["png", "jpg", "txt"])
    ]

    results = [
        item
        for sublist in [[r + f".{f}" for r in results] for f in formato_final]
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
