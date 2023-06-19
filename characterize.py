import rank
import caracteres

import os
import sys
import time
import math
import threading
import pickle
import subprocess

import argparse
import re


python = sys.executable if not "python.exe" in os.listdir(
) else os.path.join(os.path.realpath(os.path.dirname(__file__)), 'python.exe')

try:
    import numpy as np
except ImportError:
    subprocess.run(
        f"{python} -m pip install numpy")
    import numpy as np

try:
    import pathos.multiprocessing as mp
except ImportError:
    subprocess.run(
        f"{python} -m pip install pathos")
    import pathos.multiprocessing as mp
finally:
    from pathos.multiprocessing import cpu_count

try:
    from PIL import Image, ImageFont, ImageDraw, ImageEnhance
except ImportError:
    subprocess.run(
        f"{python} -m pip install Pillow-SIMD")
    from PIL import Image, ImageFont, ImageDraw, ImageEnhance


characterize_path = os.path.realpath(os.path.dirname(__file__))


def replace_many_one(string: str, lst: list, to_replace: str) -> str:
    for item in lst:
        string = string.replace(item, to_replace)
    return string


def divide_list(lst: list, n: int):
    """ Divides a list into n sublists. """
    return [list(lst[i::n]) for i in range(n)]


def amplify_differences(values, threshold):
    values = np.array(values)
    deviations = np.abs(values - threshold)
    
    amplification_factors = np.where(values >= threshold, 1 + deviations, 1 - deviations)
    amplified_values = np.clip(values * amplification_factors, 0, 1)
    
    return amplified_values


def create_char_image(caracter, color, detail, font):
    # create a new image
    new_image = Image.new('RGBA', (detail, detail), color=(0, 0, 0, 255))
    # get the font
    font = ImageFont.truetype(font, detail)
    # draw the text
    draw = ImageDraw.Draw(new_image)
    draw.fontmode = "L"
    draw.text(((detail)/2, (detail)/2), caracter, align='center', font=font,
              fill=color, anchor="mm")
    # return the new image
    return new_image


def to_hours_minutes_seconds(seconds: float):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)


def get_chars(image, lista_caracteres, diccionario_imagenes_caracteres, formato, color):
    original_width, original_height = image.size
    pixels = image.convert("L").load()
    color_levels = [[pixels[i, j]/255 for j in range(
        original_height)] for i in range(original_width)]
    # exacerbate or reduce differences to get a better b&w image
    if not color:
        color_levels_list = [
            item for sublist in color_levels for item in sublist]
        threshold = np.percentile(color_levels_list, 90)
    for i, x in enumerate(color_levels):
        color_levels[i] = list(map(lambda n: int(n * len(
            lista_caracteres) - 0.5), amplify_differences(x, threshold) if not color else x))
    # create a list
    caracteres = [[diccionario_imagenes_caracteres[lista_caracteres[y]] for y in color_int]
                  for color_int in color_levels] if any(x in formato for x in ["png", "jpg"]) else []
    caracteres_aux = np.fliplr(np.fliplr(np.array([[lista_caracteres[y] for y in color_int]
                                                   for color_int in color_levels]).transpose())) if any(x in formato for x in ["txt"]) else []
    # return the list
    return caracteres, caracteres_aux


def unite_image(caracteres, original_width, original_height, nivel_detalle_caracter):
    # create a new image
    new_image = Image.new(
        'RGBA', (nivel_detalle_caracter * original_width, nivel_detalle_caracter * original_height))
    # loop through the segments
    for i in range(original_width):
        for j in range(original_height):
            # paste the segment on the new image
            new_image.paste(caracteres[i][j],
                            (i * nivel_detalle_caracter, j * nivel_detalle_caracter))
    # return the new image
    return new_image



def divide_image(image, min_size):
    """
    Divide the image into smaller parts if its size exceeds the min_size.
    """
    image_list = [image]
    
    while image_list[0].size[0] * image_list[0].size[1] >= min_size:
        temp_list = []
        for img in image_list:
            temp_list.extend([
                img.crop((0, 0, img.width // 2, img.height // 2)),
                img.crop((img.width // 2, 0, img.width, img.height // 2)),
                img.crop((0, img.height // 2, img.width // 2, img.height)),
                img.crop((img.width // 2, img.height // 2, img.width, img.height))
            ])
        image_list = temp_list.copy()
        
    return image_list


def rutina(image, lista_caracteres, dict_imagenes_caracteres, detalle, dividir, formato, resize, color, folder_name, tkinter):
    t_image = time.time()
    image_name = "".join([x for x in image])

    # Inform if tkinter is being used
    if tkinter:
        print(f"<<{image_name}<<P>>")

    # Convert to Image object if not already
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
        
        # Resize the image if needed
        if resize[0]:
            factor_resize = max(image.size[0]/resize[1][0], image.size[1]/resize[1][1])
            image = image.resize((int(image.size[0]/factor_resize), int(image.size[1]/factor_resize)), resample=Image.Resampling.BOX)
        
        # If the image is too big, divide it into smaller parts
        if dividir:
            image_list = divide_image(image, 408960)
        else:
            image_list = [image]
    else:
        image = image.convert('RGB')
        image_list = [image]

    # Process each divided image or the whole image
    for im in image_list:
        im = ImageEnhance.Color(im).enhance(2)
        caracteres_imagen = get_chars(im, lista_caracteres, dict_imagenes_caracteres, formato, color=color)

        # If saving as text
        if "txt" in formato:
            image_name = image_name.replace("\\", "/")
            filename = os.path.join(os.path.join(characterize_path, folder_name), ''.join(image_name.split('/')[-1].split('.')[:-1]) + '.txt')
            with open(filename, "w") as f:
                f.writelines([' '.join(line) + '\n' for line in caracteres_imagen[1]])
            if tkinter:
                print(f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename}>>")

        # If saving as image
        if any(ext in formato for ext in ["png", "jpg"]):
            imagen_final = unite_image(caracteres_imagen[0], im.width, im.height, detalle)
            im = im.resize((im.width * detalle, im.height * detalle), resample=Image.Resampling.BOX)
            bg_w, bg_h = im.size
            img_w, img_h = imagen_final.size
            offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
            im.paste(imagen_final, offset, imagen_final)
            im = im.convert("RGB")
            
            # Save the image
            def save_image(image, formato, color, filename):
                if "png" in formato:
                    if color:
                        image = image.quantize(colors=256)
                    image.save(filename + ".png", compress_level=9)
                if "jpg" in formato:
                    if color:
                        image = image.quantize(colors=256)
                    image.save(filename + ".jpg")

            image_name = image_name.replace("\\", "/")
            filename = os.path.join(os.path.join(characterize_path, folder_name), ''.join(image_name.split('/')[-1].split('.')[:-1]))

            if tkinter:
                print(f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename+'.png' if 'png' in formato else filename+'.jpg'}>>")

            # Start the thread for saving the image
            thread = threading.Thread(target=save_image, args=(im, formato, color, filename), daemon=True)
            thread.start()
            thread.join()

            return filename


def choose_option(what, options_list, text=True):
    if text:
        list_as_items = [str(i+1)+') '+str(item)+'\n' for i,
                         item in enumerate(options_list)]
        items = ""
        for item in list_as_items:
            items += item
        print(
            f"\nWhich of the following {what} do you want to use?\n\n{items}")
    try:
        choice = int(input("Choice: "))
    except ValueError:
        return choose_option(what, options_list, False)
    else:
        if not 0 < choice < len(options_list)+1:
            return choose_option(what, options_list, False)
        return options_list[choice-1]


def choose_value(what, min=False, max=False, text=True):
    if text:
        print(f"\nPlease enter a {what} integer value {'between '+str(min)+' and '+str(max)+' ' if min and max else ('above '+str(min) if min else 'below '+str(max))}(float values will be converted to integers by rounding them down).\n")
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
        print("""\nPlease, input all file paths; or a directory containing them. Separate multiple paths using spaces. For paths containing spaces, use double ("") or single ('') quotes for every path.\n""")
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
    l = cl = input_files = f = tk = False
    c = d = o = cr = (False, False)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--i', type=str, nargs="*",
                        help='input file paths MANY ["path1", "path2", ...]')
    parser.add_argument('-cr',
                        '--cr', type=int, help='character resolution parameter ONE (1 to 4000)')
    parser.add_argument('-cl',
                        '--cl', type=int, help='complexity level parameter ONE (1 to 4000)')
    parser.add_argument('-l',
                        '--l', type=str, help='language parameter ONE [ascii, chinese, ...]')
    parser.add_argument('-d',
                        '--d', type=str, help='divide parameter ONE (true/false)')
    parser.add_argument('-c',
                        '--c', type=str, help='color parameter ONE (true/false)')
    parser.add_argument('-f',
                        '--f', nargs='+', help='format parameter MANY [png, jpg, txt]')
    parser.add_argument('-o',
                        '--o', type=str, help='optimize parameter ONE (true/false)')
    parser.add_argument('-tk', '--tk', type=bool,
                        help='tkinter parameter ONE (true/false)')
    args = parser.parse_args()

    input_files = args.i
    cr = (True, (args.cr, args.cr)) if isinstance(
        args.cr, int) else (False, False)
    cl = args.cl
    l = args.l

    if args.d is not None:
        d = (True, True) if any(x == args.d.lower() for x in ["true", "t", "yes", "y"]) else ((True, False) if any(x == args.d.lower() for x in ["false", "f", "no", "n"]) else (False, False))
    else:
        d = (False, False)

    if args.c is not None:
        c = (True, True) if any(x == args.c.lower() for x in ["true", "t", "yes", "y"]) else ((True, False) if any(x == args.c.lower() for x in ["false", "f", "no", "n"]) else (False, False))
    else:
        c = (False, False)

    f = " ".join(args.f) if args.f is not None else None

    if args.o is not None:
        o = (True, True) if any(x == args.o.lower() for x in ["true", "t", "yes", "y"]) else ((True, False) if any(x == args.o.lower() for x in ["false", "f", "no", "n"]) else (False, False))
    else:
        o = (False, False)

    tk = args.tk

    return l, cl, c, input_files, cr, d, f, o, tk



if __name__ == "__main__":
    if not os.path.exists(os.path.join(characterize_path, "output")):
        os.makedirs(os.path.join(characterize_path, "output"))
    if not os.path.exists(os.path.join(characterize_path, "dict_caracteres")):
        os.makedirs(os.path.join(characterize_path, "dict_caracteres"))

    fuentes = {"ascii": "arial.ttf", "arabic": "arial.ttf", "braille": "seguisym.ttf", "emoji": "seguiemj.ttf", "chinese": "msyh.ttc", "simple": "arial.ttf", "numbers+": "arial.ttf", "roman": "times.ttf", "numbers": "arial.ttf", "latin": "arial.ttf",
               "hiragana": "msyh.ttc", "katakana": "msyh.ttc", "kanji": "msyh.ttc", "cyrillic": "arial.ttf", "hangul": "malgunbd.ttf"}

    languages = list(fuentes.keys())

    idioma, nivel_complejidad, color, image_src, resize, dividir_imagen, formato_final, optimize, tkinter = parse_arguments()

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
    if not formato_final or not any(x in ["png", "jpg", "txt"] for x in formato_final.split()):
        formato_final = choose_option(
            "file formats", ["png", "jpg", "txt", "txt, png", "txt, jpg", "png, jpg", "png, jpg, txt"])
    if not optimize[0]:
        optimize = binary_choice(
            "optimize the resulting images (when images <= 300)")
    else:
        optimize = optimize[1]

    folder_name = f"output/{idioma}"

    if not os.path.exists(os.path.join(characterize_path, folder_name)):
        os.makedirs(os.path.join(characterize_path, folder_name))

    if "txt" in formato_final:
        if not os.path.exists(os.path.join(characterize_path, f"output/{idioma}/text")):
            os.makedirs(os.path.join(
                characterize_path, f"output/{idioma}/text"))

    if not any(x in formato_final for x in ("png", "jpg", "txt")):
        formato_final = "png"

    lista_imagenes = []

    for path in image_src:
        path = path.strip()
        if os.path.isdir(path):
            for image_path in [path+"/"+x for x in os.listdir(path) if any(x.lower().endswith(y) for y in [".jpg", ".jpeg", ".png", ".jfif", ".webp"])]:
                lista_imagenes.append(image_path)
        else:
            if any(path.endswith(y) for y in [".jpg", ".jpeg", ".png", ".jfif", ".webp"]):
                lista_imagenes.append(path)

    if len(lista_imagenes) == 0:
        print("No images provided. Closing...")
        sys.exit()

    nivel_detalle_caracter = 15 if idioma in [
        "hiragana", "katakana", "kanji", "chinese", "hangul", "arabic"] else (16 if idioma == "braille" else 12)

    dict_caracteres = caracteres.dict_caracteres

    try:
        font = ImageFont.truetype(fuentes[idioma], 10)
    except OSError:
        try:
            font = ImageFont.truetype("C:/Users/Augusto/Appdata/local/microsoft/windows/fonts/" +
                                      fuentes[idioma], 10)
        except OSError:
            print(
                f"{fuentes[idioma]}, the font designed for '{idioma}', can't not be found in your operating system.")
            sys.exit()
        else:
            fuente = "C:/Users/Augusto/Appdata/local/microsoft/windows/fonts/" + \
                fuentes[idioma]
    else:
        fuente = fuentes[idioma]

    t3 = time.time()

    if not f"caracteres_{idioma}-{nivel_detalle_caracter}-{nivel_complejidad}-{fuentes[idioma][0:fuentes[idioma].index('.')]}.list" in os.listdir(os.path.join(characterize_path, "dict_caracteres")):
        print("\nCreating a list containing a characters' ranking by brightness levels to accelerate the script's execution in the future...")
        lista_caracteres_original = rank.create_ranking(
            nivel_detalle_caracter, font=fuente, list_size=nivel_complejidad, allowed_characters=dict_caracteres[idioma])
        pickle.dump(lista_caracteres_original, open(os.path.join(characterize_path,
                    f"dict_caracteres/caracteres_{idioma}-{nivel_detalle_caracter}-{nivel_complejidad}-{fuentes[idioma][0:fuentes[idioma].index('.')]}.list"), "wb"))
        lista_caracteres = [x[0] for x in lista_caracteres_original]
        print(
            f"Characters list created in {round(time.time()-t3, 2)} seconds.\n")
    else:
        lista_caracteres_original = pickle.load(open(os.path.join(
            characterize_path, f"dict_caracteres/caracteres_{idioma}-{nivel_detalle_caracter}-{nivel_complejidad}-{fuentes[idioma][0:fuentes[idioma].index('.')]}.list"), "rb"))
        lista_caracteres = [x[0] for x in lista_caracteres_original]
        print(
            "\n" + f"Characters list loaded in {to_hours_minutes_seconds(time.time() -  t3)}.")

    t4 = time.time()

    if any(x in formato_final for x in ["jpg", "png"]):
        if not f"dict_caracteres_{idioma}-{nivel_detalle_caracter}-{nivel_complejidad}-{fuentes[idioma][0:fuentes[idioma].index('.')]}{'-color' if color else ''}.dictionary" in os.listdir(os.path.join(characterize_path, "dict_caracteres")):
            print(
                "Creating a dictionary containing the characters to accelerate the script's execution in the future...")
            diccionario_caracteres = {}
            for i in lista_caracteres:
                diccionario_caracteres[i] = create_char_image(
                    i, (255, 255, 255, 255) if not color else (0, 0, 0, 0), nivel_detalle_caracter, font=fuente)
            pickle.dump(diccionario_caracteres, open(os.path.join(
                characterize_path, f"dict_caracteres/dict_caracteres_{idioma}-{nivel_detalle_caracter}-{nivel_complejidad}-{fuentes[idioma][0:fuentes[idioma].index('.')]}{'-color' if color else ''}.dictionary"), 'wb'))
            print(
                f"Characters dict created in {round(time.time()-t4, 2)} seconds.\n")
        else:
            diccionario_caracteres = pickle.load(open(os.path.join(
                characterize_path, f"dict_caracteres/dict_caracteres_{idioma}-{nivel_detalle_caracter}-{nivel_complejidad}-{fuentes[idioma][0:fuentes[idioma].index('.')]}{'-color' if color else ''}.dictionary"), 'rb'))
            print(
                f"Characters dict loaded in {to_hours_minutes_seconds(time.time() -  t4)}.\n")
    else:
        diccionario_caracteres = False

    t = cpu_count() if cpu_count() >= 2 else 1
    num_iterations = len(lista_imagenes)

    if len(lista_caracteres) <= 30:
        print("Characters to use:", lista_caracteres_original, "\n")
    print(
        f"Processing {num_iterations} {'image' if num_iterations == 1 else 'images'}{' in '+str(math.ceil(num_iterations/t))+' cycles' if not t > num_iterations else ''}...", end="\n\n")

    t0 = time.time()

    if num_iterations == 1:
        t_interno = time.time()
        results = []
        for i, image in enumerate(lista_imagenes):
            results.append(rutina(image, lista_caracteres, diccionario_caracteres,
                                  nivel_detalle_caracter, dividir=dividir_imagen, formato=formato_final, resize=resize, color=color, folder_name=folder_name, tkinter=tkinter))
        if not tkinter:
            print(
                f"Elapsed time: {to_hours_minutes_seconds(time.time()-t_interno)}")
    else:
        t = t if t <= num_iterations else num_iterations
        results = []

        cycles = math.ceil(num_iterations/t)

        l = 0
        m = 0
        n = t

        pool = mp.ProcessingPool(t)

        while num_iterations > 0:
            t_interno = time.time()
            iterations = t
            if num_iterations < t:
                iterations = num_iterations
                pool = mp.ProcessingPool(iterations)
            value_range = [m+k for k in range(iterations)]
            results += pool.imap(rutina, [lista_imagenes[k] for k in value_range], [lista_caracteres for _ in value_range], [diccionario_caracteres for _ in value_range],
                                 [nivel_detalle_caracter for _ in value_range], [dividir_imagen for _ in value_range], [formato_final for _ in value_range], [resize for _ in value_range], [color for _ in value_range], [folder_name for _ in value_range], [tkinter for _ in value_range])
            num_iterations -= iterations
            l += 1
            m += iterations
            n += iterations
            if not tkinter:
                print(f"   {l}/{cycles} - Total execution time: {to_hours_minutes_seconds(round((time.time() - t0), 2))} ({to_hours_minutes_seconds(time.time() - t_interno)}) / {to_hours_minutes_seconds(round((time.time() - t_interno) * (cycles-l), 2))} remaining     ", end="\r")

    formato_final = [replace_many_one(x, [",", ".", ";"], "")
                     for x in [y for y in formato_final.split() if any(z in y for z in ["png", "jpg", "txt"])]]

    results = [item for sublist in [
        [r+f".{f}" for r in results] for f in formato_final] for item in sublist]

    if optimize and len(results) <= 300:
        if os.path.exists("C:/Program Files/FileOptimizer/FileOptimizer64.exe"):
            print("\n\nOptimizing files in the background...")
            if len(results) >= t//2:
                paths = divide_list(results, t//2)
                for divided in paths:
                    divided_str = str([f'"{x}"' for x in divided]).replace(
                        ", ", " ")[1:-1].replace("'", "")
                    subprocess.Popen(
                        f'C:/Program^ Files/FileOptimizer/FileOptimizer64.exe {divided_str}', shell=True)
            else:
                paths = divide_list(results, len(results))
                for divided in paths:
                    divided_str = str([f'"{x}"' for x in divided]).replace(
                        ", ", " ")[1:-1].replace("'", "")
                    subprocess.Popen(
                        f'C:/Program^ Files/FileOptimizer/FileOptimizer64.exe {divided_str}', shell=True)
    else:
        print("\n\nBypassing file optimization...")

    print(
        "\n"+f"All done. Characterized images can be found in {os.path.join(characterize_path, folder_name)}.")
