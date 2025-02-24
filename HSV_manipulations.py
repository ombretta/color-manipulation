import os

import cv2
import matplotlib.pyplot as plt

import numpy as np

from utils import print_image, stitch_plots


def get_HSV_RGB(img_file, scale_factor=1.0):
    """
    Reads an image file, converts it from BGR to RGB, and returns its HSV and RGB representations.

    Parameters:
    -----------
    img_file : str
        Path to the image file.
    scale_factor : float, optional (default=1.0)
        Scaling factor for downsampling the image.

    Returns:
    --------
    hsv_img : numpy.ndarray
        The image converted to HSV color space.
    rgb_img : numpy.ndarray
        The image in RGB color space.
    """
    bgr_img = cv2.imread(img_file)
    print(img_file)
    rgb_img = bgr_img[:, :, ::-1]

    downscaled_bgr_img = cv2.resize(bgr_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    hsv_img = cv2.cvtColor(downscaled_bgr_img, cv2.COLOR_BGR2HSV)

    return hsv_img, rgb_img


def plot_colorwheel(x, y, colors, path, figsize=(10, 10)):
    """Plots the color wheel."""
    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=colors / 255)

    # Draw circle boundary
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = 255 * np.cos(theta)
    y_circle = 255 * np.sin(theta)

    plt.scatter(x_circle, y_circle, c=np.zeros((len(x_circle), 3)), s=0.1)
    plt.xlim([-255, 255])
    plt.ylim([-255, 255])
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.axis("off")

    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_color_wheel(hsv, rgb, colors, examples_dir, img_file):
    """Generates and saves a color wheel visualization."""
    theta = hsv[:, 0].astype('float16') * 2 * np.pi / 180.0  # Convert hue to radians
    r = hsv[:, 1]  # Saturation as radius
    x_coordinates = r * np.cos(theta)
    y_coordinates = r * np.sin(theta)

    if examples_dir and not os.path.exists(examples_dir + img_file.replace(".jpg", ".png")):
        print_image(rgb, examples_dir + "RGB.png")
        plot_colorwheel(x_coordinates, y_coordinates, colors, path=examples_dir + "colorwheel.png")
        stitch_plots(examples_dir + "RGB.png", examples_dir + "colorwheel.png",
                     examples_dir + img_file.replace(".jpg", ".png"))
        os.remove(examples_dir + "RGB.png")
        os.remove(examples_dir + "colorwheel.png")


def print_new_image(HSV_new, downsampling_factor, file_dir, file_name):
    """Converts HSV image to RGB and generates a color wheel visualization."""
    rgb_image = cv2.cvtColor(HSV_new, cv2.COLOR_HSV2RGB)
    colors = cv2.cvtColor(HSV_new, cv2.COLOR_HSV2BGR).reshape(-1, 3)[::downsampling_factor][:, ::-1]
    HSV = HSV_new.transpose(2, 0, 1).reshape((3, -1)).transpose(1, 0)[::downsampling_factor]
    plot_color_wheel(HSV, rgb_image, colors, file_dir, file_name)


def rotate_hue(HSV, degrees, downsampling_factor, img_file, result_dir):

    """
    Rotates the hue channel of an HSV image by a specified number of degrees. The function modifies the hue channel
    in place and saves the updated image using `print_new_image`.

    Parameters:
    -----------
    HSV : numpy.ndarray
        A 3D NumPy array representing the image in HSV color space, where
        HSV[:, :, 0] corresponds to the hue channel.
    degrees : int
        The number of degrees to rotate the hue values.
    downsampling_factor : int
        The factor by which the image should be downsampled before saving.
    img_file : str
        The filename of the input image, used for naming the output file.
    result_dir : str
        The directory where the modified image will be saved.

    Returns the modified image file in RGB format.

    Notes:
    ------
    - The hue values in the HSV image are scaled between 0 and 180 in OpenCV.
    - The function shifts the hue values by `degrees`, ensuring they remain within the valid range.
    - The output filename is modified to reflect the applied hue rotation.
    """

    theta = HSV[:, :, 0].astype('float16') * 2  # Angle in degrees
    th = ((theta + degrees) % 360) / 2
    HSV[:, :, 0] = th
    img_file = img_file.replace(".jpg", f"_{degrees}degrees.jpg")
    cv2.imwrite(result_dir+img_file, cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR))
    img_file = img_file.replace(".jpg", "_colorwheel.jpg")
    print_new_image(HSV, downsampling_factor, result_dir, img_file)
    return cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)


def change_saturation(HSV, delta, downsampling_factor, img_file, result_dir):

    """Saturation values range from 0 to 255. The parameter 'delta' is the delta between the original saturation values
    and the modified saturation values. It can be positive or negative."""

    saturation = HSV[:, :, 1].astype('float16')
    HSV[:, :, 1] = np.maximum(0, np.minimum(saturation + delta, 255))
    file_name = img_file.replace(".jpg", f"_{delta}saturation.jpg")
    print_new_image(HSV, downsampling_factor, result_dir, file_name)


def change_brightness(HSV, delta, downsampling_factor, img_file, result_dir):

    """Brightness (or value) values range from 0 to 255. The parameter 'delta' is the delta between the original
    brightness values and the modified brightness values. It can be positive or negative."""

    brightness = HSV[:, :, 2].astype('float16')
    HSV[:, :, 2] = np.maximum(0, np.minimum(brightness + delta, 255))
    file_name = img_file.replace(".jpg", f"_{delta}brightness.jpg")
    print_new_image(HSV, downsampling_factor, result_dir, file_name)
    return HSV


def get_HSV_parameters(img_file, image_dir, examples_dir=None, downsampling_factor=10):
    hsv_img, rgb_img = get_HSV_RGB(image_dir+img_file, scale_factor=0.1)

    # Downsample pixels for efficiency
    hsv_downsampled = hsv_img.transpose(2, 0, 1).reshape((3, -1)).transpose(1, 0)[::downsampling_factor]

    # Define the colors on the color wheel
    colors = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR).reshape(-1, 3)[::downsampling_factor][:, ::-1]

    avg_hue, avg_hue_distance, std_hue_distance, avg_sat, std_sat, avg_bright, std_bright = (
        plot_color_wheel(hsv_downsampled, rgb_img, colors, examples_dir, img_file))

    return avg_hue, avg_hue_distance, std_hue_distance, avg_sat, std_sat, avg_bright, std_bright