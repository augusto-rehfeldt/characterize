import os
import time
from PIL import Image, ImageEnhance, UnidentifiedImageError
import numpy as np

from .image_utils import divide_image, get_chars, unite_image, save_image, save_text

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

    if tkinter:
        print(f"<<{image_name}<<P>>")

    # Load or convert image
    if not isinstance(image, Image.Image):
        original_path = image
        try:
            image = Image.open(image).convert("RGB")
        except UnidentifiedImageError:
            print(f"Warning: Cannot identify image file '{original_path}'. Skipping.")
            return None
        # Resize if requested
        if resize[0]:
            factor = max(image.size[0] / resize[1][0], image.size[1] / resize[1][1])
            image = image.resize(
                (int(image.size[0] / factor), int(image.size[1] / factor)),
                resample=Image.Resampling.LANCZOS
            )
        # Divide if large
        if divide_image_flag:
            image_list = divide_image(image, 408960)
        else:
            image_list = [image]
    else:
        image = image.convert("RGB")
        image_list = [image]

    # Process each segment
    for im in image_list:
        im = ImageEnhance.Color(im).enhance(2)
        chars_img, chars_txt = get_chars(im, character_list, char_images, output_format, color=color)

        if "txt" in output_format:
            name = image_name.replace("\\", "/")
            filename = os.path.join(
                os.path.join(os.path.dirname(__file__), os.pardir, folder_name),
                "".join(name.split("/")[-1].split(".")[:-1]) + ".txt",
            )
            save_text(chars_txt, filename)
            if tkinter:
                print(f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename}>>")

        if any(ext in output_format for ext in ("png", "jpg")):
            final_np = unite_image(chars_img, im.width, im.height, character_detail_level)
            im_resized = im.resize(
                (im.width * character_detail_level, im.height * character_detail_level),
                resample=Image.Resampling.BOX
            )
            bg = np.array(im_resized.convert("RGBA"))
            alpha = final_np[:, :, 3:4] / 255.0
            # Determine paste position (center)
            bh, bw = bg.shape[:2]
            ih, iw = final_np.shape[:2]
            y_off = (bh - ih) // 2
            x_off = (bw - iw) // 2
            # Composite
            slice_bg = bg[y_off:y_off+ih, x_off:x_off+iw]
            slice_comb = final_np[:, :, :3] * alpha + slice_bg[:, :, :3] * (1 - alpha)
            bg[y_off:y_off+ih, x_off:x_off+iw, :3] = slice_comb.astype(np.uint8)
            combined = Image.fromarray(bg, 'RGBA').convert("RGB")
            if color:
                combined = combined.quantize(colors=256)
            else:
                combined = combined.convert("L")
            name = image_name.replace("\\", "/")
            filename = os.path.join(
                os.path.join(os.path.dirname(__file__), os.pardir, folder_name),
                "".join(name.split("/")[-1].split(".")[:-1]),
            )
            save_image(combined, output_format, color, filename)
            if tkinter:
                ext = ".png" if "png" in output_format else ".jpg"
                print(f"<<{image_name}<<{round(time.time()-t_image, 2)}<<{filename+ext}>>")
            return filename

def process_image(image, character_list, char_images,
                  character_detail_level, divide_image_flag,
                  output_format, resize, color, folder_name, tkinter):
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
