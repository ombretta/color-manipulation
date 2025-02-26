import os
from tqdm import tqdm
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colorthief import ColorThief
from PIL import Image, UnidentifiedImageError


def get_palette(image_path, n=6):
    """
    Get the color palette of the image using the ColorThief library.
    Based on clustering + counting.
    https://github.com/fengsp/color-thief-py/blob/master/colorthief.py
    """

    Image.MAX_IMAGE_PIXELS = 500000000

    try:
        img = Image.open(image_path)

        # Resize the image to a smaller size for efficiency
        max_width = 400
        new_width = min(img.size[0], max_width)
        aspect_ratio = float(img.size[1]) / float(img.size[0])
        new_height = int(new_width * aspect_ratio)
        img = img.resize((new_width, new_height))
        img.save("resized_image.jpg")

        color_thief = ColorThief("resized_image.jpg")
        palette = color_thief.get_palette(color_count=n, quality=1)
        palette = np.array([palette])

        os.remove("resized_image.jpg")

        return palette

    except UnidentifiedImageError:
        print("Could not open ", image_path)
        os.remove(image_path)
        return None


def extract_image_palettes(image_path, savedir_feature, savedir_image, n=6):
    # Extract color palette
    palette = get_palette(image_path, n)

    # Save color palette
    if palette is not None:
        # Save features
        image_name = image_path.split("/")[-1]
        os.makedirs(savedir_feature, exist_ok=True)
        with open(savedir_feature + image_name.replace(".jpg", ".npy"), 'wb') as f:
            np.save(f, palette)

        # Save palette as image
        os.makedirs(savedir_image, exist_ok=True)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(palette.shape[1], palette.shape[0])
        plt.imshow(palette, aspect='auto')
        plt.grid(False)
        plt.axis("off")
        plt.savefig(savedir_image + image_name.replace(".jpg", f"_{n}d.png"), bbox_inches='tight', dpi=100)

    return palette


def nearest(palette, color):
    """From https://github.com/fengsp/color-thief-py/blob/master/colorthief.py"""
    d1 = None
    p_color = None
    for i in range(len(palette)):
        d2 = math.sqrt(
            math.pow(color[0] - palette[i][0], 2) +
            math.pow(color[1] - palette[i][1], 2) +
            math.pow(color[2] - palette[i][2], 2)
        )
        if d1 is None or d2 < d1:
            d1 = d2
            p_color = palette[i]
    return p_color


def nearest_recoloring(palette1, palette2, color):
    """Recolor an image according to the palette of another image.
    From https://github.com/fengsp/color-thief-py/blob/master/colorthief.py"""
    d1 = None
    p_color = None
    for i in range(len(palette1)):
        d2 = math.sqrt(
            math.pow(color[0] - palette1[i][0], 2) +
            math.pow(color[1] - palette1[i][1], 2) +
            math.pow(color[2] - palette1[i][2], 2)
        )
        if d1 is None or d2 < d1:
            d1 = d2
            p_color = palette2[i]
    return p_color


def map_image_to_palette(image_path, savedir_image, n=6):
    """This function simplifies the colors of an image by mapping all its pixels to the dominant palette."""

    # Open the image and get the pixel values
    img = np.asarray(Image.open(image_path))  # (600, 432, 3)
    pixels = img.reshape(-1, 3)  # (259200, 3)

    # Extract the n-dimensional dominant palette
    palette = get_palette(image_path, n)
    palette = np.squeeze(palette, axis=0)

    # Map all the pixels to the dominant palette
    new_pixels = np.apply_along_axis(lambda x: nearest(palette, x), axis=1, arr=pixels)
    new_image = new_pixels.reshape(img.shape)

    # Save the new image
    image_name = image_path.split("/")[-1]
    os.makedirs(savedir_image, exist_ok=True)
    cv2.imwrite(savedir_image + image_name + f"_simplified_{n}d.jpg", new_image[:, :, ::-1])


def image_recoloring(image_path, color_path, savedir_image, n=6):
    """This function recolors an image with the palette of a different image.

    image_path: path of the original image.
    color_path: path of the image that will give the new palette.
    savedir_image: where to save the resulting image.
    n: number of colors in the palettes."""

    # Open the image and get the pixel values
    img = np.asarray(Image.open(image_path))
    pixels = img.reshape(-1, 3)

    # Extract the n-dimensional dominant palette
    palette_original = np.squeeze(get_palette(image_path, n), axis=0)
    palette_new = np.squeeze(get_palette(color_path, n), axis=0)

    # Map all the pixels to the new palette
    new_pixels = np.apply_along_axis(lambda x: nearest_recoloring(palette_original, palette_new, x), axis=1, arr=pixels)
    new_image = new_pixels.reshape(img.shape)

    # Save the new image
    image_name = image_path.split("/")[-1]
    os.makedirs(savedir_image, exist_ok=True)
    cv2.imwrite(savedir_image + image_name + f"_recolored_{n}d.jpg", new_image[:, :, ::-1])


def main():

    n = 20
    savedir_feature = f"results/color_palettes/examples/{n}/"
    original_image_path = "images/adnan-coker_unknown-title(3).jpg"
    # original_image_path = "images/nicholas-roerich_tent-mountain-1933.jpg"
    # new_color_path = "images/ad-reinhardt_yellow-painting-1949.jpg"
    new_color_path = "images/nicholas-roerich_tent-mountain-1933.jpg"
    # new_color_path = "images/adnan-coker_unknown-title(3).jpg"
    results_path = "results/image_recoloring/change_palette/examples/"
    # extract_image_palettes(original_image_path, savedir_feature, results_path, n)
    # map_image_to_palette(original_image_path, results_path, n)
    image_recoloring(original_image_path, new_color_path, results_path, n)


if __name__ == "__main__":
    main()