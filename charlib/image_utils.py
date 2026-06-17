import numpy as np
from PIL import Image, ImageFont, ImageDraw

try:
    import cv2
except ImportError:  # Optional for terminal-only / image-only usage
    cv2 = None

# ANSI escape code constants
ANSI_RESET = "\033[0m"
ANSI_FOREGROUND = "\033[38;5;{}m"

def rgb_to_ansi256(r, g, b):
    """Converts RGB to the nearest ANSI 256 color code."""
    if r == g == b: # Grayscale
        if r < 8: return 16
        if r > 248: return 231
        return round(((r - 8) / 247) * 24) + 232
    # Color
    return (16 + (36 * round(r / 255 * 5)) +
            (6 * round(g / 255 * 5)) +
            round(b / 255 * 5))

def amplify_differences(values, threshold):
    arr = np.array(values)
    deviations = np.abs(arr - threshold)
    factors = np.where(arr >= threshold, 1 + deviations, 1 - deviations)
    amplified = np.clip(arr * factors, 0, 1)
    return amplified

def create_char_image(char, color, detail, font_path):
    img = Image.new("RGBA", (detail, detail), color=(0, 0, 0, 255))
    font = ImageFont.truetype(font_path, detail)
    draw = ImageDraw.Draw(img)
    draw.fontmode = "L"
    draw.text((detail/2, detail/2), char, font=font, fill=color, anchor="mm", align="center")
    return np.array(img)

def create_char_image_dict(characters, detail, font_path, color=False):
    images = {}
    fill = (255, 255, 255, 255) if not color else (0, 0, 0, 0)
    for ch in characters:
        images[ch] = create_char_image(ch, fill, detail, font_path)
    return images

def get_chars(image, char_list, char_images, fmt, color):
    # Always use grayscale for character selection based on intensity
    gray_img = image.convert("L")
    gray_data = np.array(gray_img) / 255.0

    # Use amplification only if not in color mode (to preserve original brightness mapping)
    if not color:
        threshold = np.percentile(gray_data, 90)
        intensity_map = amplify_differences(gray_data, threshold)
    else:
        intensity_map = gray_data

    indices = (intensity_map * len(char_list) - 0.5).astype(int)
    indices = np.clip(indices, 0, len(char_list) - 1)

    out_img = []
    if any(x in fmt for x in ("png", "jpg")):
        # Image output generation (unchanged for now)
        out_img = [[char_images[char_list[ix]] for ix in row] for row in indices.tolist()]

    out_txt = []
    if "txt" in fmt:
        char_array = np.array(char_list)
        mapped_chars = char_array[indices]

        if color:
            # Get RGB data if color output is needed
            rgb_img = image.convert("RGB")
            rgb_data = np.array(rgb_img)
            h, w = mapped_chars.shape
            colored_txt_rows = []
            for r in range(h):
                row_str = []
                for c in range(w):
                    char = mapped_chars[r, c]
                    rgb = rgb_data[r, c]
                    ansi_code = rgb_to_ansi256(rgb[0], rgb[1], rgb[2])
                    row_str.append(f"{ANSI_FOREGROUND.format(ansi_code)}{char}{ANSI_RESET}")
                colored_txt_rows.append(row_str)
            # Transpose and flip like before? Let's test without first.
            # The original flip was likely due to how numpy arrays vs terminal output work.
            # Let's try direct output first. If it's mirrored, we'll re-add flips.
            out_txt = colored_txt_rows # Keep original orientation for now
        else:
            # Original grayscale text generation
            # flipped = np.fliplr(np.fliplr(mapped_chars.transpose())) # Original flip
            # out_txt = flipped.tolist()
            out_txt = mapped_chars.tolist() # Keep original orientation

    return out_img, out_txt

def unite_image(chars, orig_w, orig_h, detail):
    """
    Efficiently combines small character images into a single large image using numpy.
    """
    # Convert the list of lists of images into a single 4D numpy array
    # The shape will be (orig_h, orig_w, detail, detail, 4)
    try:
        char_array = np.array(chars, dtype=np.uint8)
    except ValueError as e:
        print(f"Error creating numpy array from chars. This can happen if the character images are not all the same size. Details: {e}")
        # Fallback to a slower, more robust method if the fast path fails
        h = orig_h * detail
        w = orig_w * detail
        final = np.zeros((h, w, 4), dtype=np.uint8)
        for j in range(orig_h):
            for i in range(orig_w):
                cell = chars[j][i]
                y0, y1 = j * detail, (j + 1) * detail
                x0, x1 = i * detail, (i + 1) * detail
                try:
                    # Ensure the cell has the expected dimensions before slicing
                    if cell.shape[0] >= detail and cell.shape[1] >= detail:
                        final[y0:y1, x0:x1] = cell[:detail, :detail, :]
                    else:
                        # Handle cases where the cell is smaller than expected
                        # Create a black patch and place the smaller cell in it
                        patch = np.zeros((detail, detail, 4), dtype=np.uint8)
                        h_cell, w_cell, _ = cell.shape
                        patch[:h_cell, :w_cell, :] = cell
                        final[y0:y1, x0:x1] = patch
                except Exception as ex:
                    print(f"Warning: Cell at ({i},{j}) with shape {cell.shape} could not be placed. Error: {ex}")
        return final

    # ponytail: create_char_image always yields (detail, detail, 4), so ndim==5 holds; no fallback branch needed.

    # Reshape and transpose to combine the character images
    # 1. Transpose to (orig_h, detail, orig_w, detail, 4)
    # 2. Reshape to (orig_h * detail, orig_w * detail, 4)
    final_image = char_array.transpose(0, 2, 1, 3, 4).reshape(orig_h * detail, orig_w * detail, 4)
    
    return final_image

def sample_colors(img, width, height):
    """
    Efficiently samples colors from an image by resizing it to the target dimensions.
    """
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    if cv2 is not None:
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = np.asarray(img)

        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        return cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)

    # Fallback path for environments without OpenCV.
    if isinstance(img, Image.Image):
        pil_img = img.convert("RGB")
    else:
        img_np = np.asarray(img)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        pil_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), "RGB")

    resized_img = pil_img.resize((width, height), resample=Image.Resampling.BOX)
    return np.array(resized_img)

def divide_image(image, min_size):
    parts = [image]
    while parts and parts[0].size[0] * parts[0].size[1] >= min_size:
        new_parts = []
        for img in parts:
            w, h = img.width, img.height
            new_parts.extend([
                img.crop((0, 0, w//2, h//2)),
                img.crop((w//2, 0, w, h//2)),
                img.crop((0, h//2, w//2, h)),
                img.crop((w//2, h//2, w, h)),
            ])
        parts = new_parts
    return parts

def save_image(image, fmt, color, filename):
    opts = {
        "png": {"format": "PNG", "compress_level": 9},
        "jpg": {"format": "JPEG", "quality": 95},
    }
    for ext in ("png", "jpg"):
        if ext in fmt:
            try:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if color and ext == "png":
                    image = image.quantize(colors=256)
                image.save(f"{filename}.{ext}", **opts[ext])
            except Exception as e:
                print(f"Failed to save {filename}.{ext}: {e}")
                raise

def save_text(chars, filename, color=False):
    # Note: Saving color text with ANSI codes to a file might not be ideal
    # for viewing outside terminals that support them.
    with open(filename, "w", encoding='utf-8') as f: # Use utf-8 for wider char support
        for line in chars:
            # If color is enabled, chars already contain ANSI codes
            f.write("".join(line) + "\n")
