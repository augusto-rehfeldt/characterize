import sys

from PIL import Image

from .image_utils import get_chars, sample_colors


def render_terminal_image(image, char_list, width, height, color=False, true_color=False, clear=True, header=None):
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")

    resized_image = pil_image.resize((width, height))
    color_grid = None

    if color or true_color:
        source_for_color = pil_image if true_color else resized_image
        color_grid = sample_colors(source_for_color, width, height)

    _, text_grid = get_chars(resized_image, char_list, None, fmt="txt", color=False)

    if clear:
        sys.stdout.write("\x1b[2J\x1b[H")

    if header:
        sys.stdout.write(f"{header}\n")

    for y, line in enumerate(text_grid):
        rendered_line = []
        for x, char_val in enumerate(line):
            if color_grid is not None:
                r, g, b = color_grid[y, x]
                rendered_line.append(f"\033[38;2;{r};{g};{b}m{char_val}")
            else:
                rendered_line.append(char_val)
        sys.stdout.write("".join(rendered_line) + "\033[0m\n")

    sys.stdout.flush()
