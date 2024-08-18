import math
import sys
import os
import subprocess


python = (
    sys.executable
    if not "python.exe" in os.listdir()
    else os.path.join(os.path.realpath(os.path.dirname(__file__)), "python.exe")
)

try:
    import numpy as np
except ImportError:
    import shlex
    import subprocess
    
    python = "python3"
    subprocess.run([python, "-m", "pip", "install", "numpy"])
    import numpy as np

try:
    from PIL import Image, ImageFont, ImageDraw
except ImportError:
    import subprocess
    
    subprocess.run([python, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageFont, ImageDraw


import subprocess

def diss_index(characters):
    if isinstance(characters[0], tuple):
        min_color_level = characters[0][1]
        max_color_level = characters[-1][1]
        step_sizes = [
            abs(color1 - color2)
            for (_, color1), (_, color2) in zip(characters[:-1], characters[1:])
        ]
        try:
            median_step_size = np.median(step_sizes)
        except AttributeError:
            subprocess.run(f&quot;{python} -m pip install --upgrade numpy&quot;, shell=False)
            median_step_size = np.median(step_sizes)
        dissimilarity_index = round(
            median_step_size / (max_color_level - min_color_level), 3
        )
        return dissimilarity_index
    else:
        min_color_level = characters[0]
        max_color_level = characters[-1]
        step_sizes = [
            abs(color1 - color2)
            for color1, color2 in zip(characters[:-1], characters[1:])
        ]
        try:
            median_step_size = np.median(step_sizes)
        except AttributeError:
            subprocess.run(f&quot;{python} -m pip install --upgrade numpy&quot;, shell=False)
            median_step_size = np.median(step_sizes)
        dissimilarity_index = round(
            median_step_size / (max_color_level - min_color_level), 3
        )
        return dissimilarity_index


def decimal_range(start, stop, length):
    color_range = stop - start
    step_size = math.ceil(color_range / length)
    values = [start + step_size * x for x in range(0, length)]
    return values


def char_image_colors(character, detail, font):
    # create a new image
    new_image = Image.new("L", (detail, detail), color=0)
    # get the font
    font = ImageFont.truetype(font, detail)
    # draw the text
    draw = ImageDraw.Draw(new_image)
    draw.fontmode = "L"
    
    # Special handling for empty character
    if character.strip() == "":
        # Return the empty character with brightness 0
        return (character, 0)
    
    draw.text(
        ((detail) / 2, (detail) / 2),
        character,
        align="center",
        font=font,
        fill=255,
        anchor="mm",
    )
    # return the new image with the average color level
    return (character, sum(new_image.getdata()) / new_image.width / new_image.height)

def choose_characters(characters, distance_values):
    characters_bis = characters.copy()
    # Initialize the list of selected characters
    selected_characters = []

    # Iterate over the distance values
    for d in distance_values:
        # Find the closest character to the current distance value
        closest_char = None
        closest_diff = float("inf")
        for char, color in characters_bis:
            diff = abs(color - d)
            if diff < closest_diff:
                closest_char = char
                closest_color = color
                closest_diff = diff

        # Add the closest character to the list of selected characters
        selected_characters.append((closest_char, closest_color))

        characters_bis.remove((closest_char, closest_color))

    # Ensure the empty character is included if it's in the original list
    if (" ", 0) in characters and (" ", 0) not in selected_characters:
        selected_characters.append((" ", 0))

    selected_characters = sorted(selected_characters, key=lambda x: x[1])

    return selected_characters

def filter_ranking(rank_list, selected_list_size):
    if len(rank_list) < 1:
        print("Error on ranking filtering. Characters list is empty.")
        sys.exit()

    if len(rank_list) < selected_list_size:
        selected_list_size = len(rank_list)

    # Check if empty string is present and handle it separately
    empty_char = None
    if (" ", 0) in rank_list:
        empty_char = (" ", 0)
        rank_list = [char for char in rank_list if char != (" ", 0)]
        selected_list_size -= 1  # Reduce size by 1 to account for empty char

    distance_values = decimal_range(
        int(rank_list[0][1]), int(rank_list[-1][1]), selected_list_size
    )

    # Choose the characters that have the closest color values to distance values
    selected_characters = choose_characters(rank_list, distance_values)

    # Add empty character back if it was present
    if empty_char:
        selected_characters = [empty_char] + selected_characters

    # calculate median step size
    dissimilarity_index = diss_index(selected_characters)

    # Calculate brightness range accounting for empty string
    min_brightness = min(char[1] for char in selected_characters)
    max_brightness = max(char[1] for char in selected_characters)
    brightness_range = int(max_brightness + 0.5) - int(min_brightness + 0.5)

    print(
        f"\nBrightness range: {brightness_range}; min and max: ({int(min_brightness + 0.5)}, {int(max_brightness + 0.5)}); dissimilarity: {dissimilarity_index}.\n"
    )

    return selected_characters

def create_ranking(
    detail,
    font,
    list_size=12,
    allowed_characters="0123456789abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ",
    include_empty=False
):
    if include_empty:
        allowed_characters += " "
    
    ranking = [
        (y[0], y[1])
        for y in sorted(
            [char_image_colors(x, detail, font) for x in allowed_characters],
            key=lambda z: z[1],
        )
    ]
    ranking = filter_ranking(ranking, list_size)
    return ranking