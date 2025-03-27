import argparse
import os

import numpy as np
import cv2

from HSV_manipulations import get_HSV_RGB, rotate_hue, change_saturation, change_brightness, plot_colorwheel
from utils import print_image, imshow

def angular_distance(t1, t2):
    return min(abs(t1 - t2), abs(t1 - 360 - t2), abs(t1 - (t2 - 360)))

def map_pixel(hue, axis_degree, rescale_shift=1):

    axis_angle1 = axis_degree%360
    axis_angle2 = (axis_degree+180)%360

    d1 = angular_distance(hue, axis_angle1)
    d2 = angular_distance(hue, axis_angle2)
    direction = np.argmin((d1, d2))

    shift = (1-direction)*angular_distance(axis_angle1, hue) + direction*angular_distance(axis_angle2, hue)
    shift /= rescale_shift

    if (angular_distance(hue+shift, [axis_angle1, axis_angle2][direction]) <
            angular_distance(hue-shift, [axis_angle1, axis_angle2][direction])):
        angle_shift = shift
    else:
        angle_shift = -shift

    return (hue + angle_shift)%360

def plot_color_axis(hue_degree, results_dir, n_pixels=30):
    """Plot the color axis to which we wish to map the image pixels.
    hue_degree: degree for one of the two hues of the axis. For orange (and teal), 30.
    n_pixels: how many pixels to plot per hue."""

    # Saturation array (distance to the center of the color cirle)
    r = np.linspace(0, 255, num=n_pixels, dtype=int)

    # Define pixels of color 1
    theta1 = (hue_degree%360) * np.ones(n_pixels) * np.pi / 180.0
    x_coordinates1 = r * np.cos(theta1)
    y_coordinates1 = r * np.sin(theta1)

    # Define pixels of color 2
    theta2 = ((hue_degree+180)%360) * np.ones(n_pixels) * np.pi / 180.0
    x_coordinates2 = r * np.cos(theta2)
    y_coordinates2 = r * np.sin(theta2)

    # Extract the position on the cirle
    x_coordinates = np.concatenate((x_coordinates1, x_coordinates2))
    y_coordinates = np.concatenate((y_coordinates1, y_coordinates2))

    # Compose h, s, v values
    h = np.expand_dims(np.concatenate((theta1, theta2), axis=0) * 180 / (2 * np.pi), axis=1)
    s = np.expand_dims(np.concatenate((r, r), axis=0), axis=1)
    v = np.ones((len(h), 1)) * 255

    # Define colors for the plot
    hsv_pixels = np.concatenate((h, s, v), axis=1).astype(np.uint8)
    hsv_pixels = hsv_pixels.reshape((1, -1, 3))
    colors = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2RGB).reshape((-1, 3))

    # Plot the colorwheel
    plot_colorwheel(x_coordinates, y_coordinates, colors, path=results_dir + "color_axis.jpg")
    colorwheel = cv2.imread(results_dir + "color_axis.jpg")[:, :, ::-1]
    imshow(colorwheel)


def plot_color_wheel(HSV_img, result_path, downsampling_factor=1):
    hsv_downsampled = HSV_img.transpose(2, 0, 1).reshape((3, -1)).transpose(1, 0)[::downsampling_factor]

    # Define the colors on the color wheel
    colors = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR).reshape(-1, 3)[::downsampling_factor][:, ::-1]

    theta = hsv_downsampled[:, 0].astype('float16') * 2 * np.pi / 180.0  # Convert hue to radians
    r = hsv_downsampled[:, 1]  # Saturation as radius
    x_coordinates = r * np.cos(theta)
    y_coordinates = r * np.sin(theta)

    plot_colorwheel(x_coordinates, y_coordinates, colors, path=result_path)
    colorwheel = cv2.imread(result_path)[:, :, ::-1]
    imshow(colorwheel)


def recolor_image(hsv_img, hue_degree, rescale_shift, img_file, results_dir):
    """
    Map the color of an image to a color axis, e.g., orange-teal.
    hsv_img: the image in hsv color space.
    hue_degree: degree for one of the two hues of the axis. For orange (and teal), 30.
    rescale_shift: determines the extent of the color mapping. If 1, the pixels are mapped exactly on the axis. If > 1,
    the pixels are mapped closer (but not onto) the axis.
    img_file: original image file.
    results_dir: where to save the results.
    """

    # Save height and width
    height, width = hsv_img.shape[:2]

    # Reshape image and extract values for hues (H)
    hsv_img = hsv_img.astype(float) # If you remove this all calculations are messed up
    HSV_img_pixels = hsv_img.transpose(2, 0, 1).reshape((3, -1)).transpose(1, 0)

    H = HSV_img_pixels[:, 0] * 2  # Hue in degrees

    # Map hues to color axis
    H = np.array(list(map(lambda x: map_pixel(x, hue_degree, rescale_shift=rescale_shift), H)))

    # Reconstruct image
    HSV_img_pixels = np.concatenate((np.expand_dims((H / 2), axis=1),
                                     np.expand_dims(HSV_img_pixels[:, 1], axis=1),
                                     np.expand_dims(HSV_img_pixels[:, 2], axis=1)), axis=1).astype(np.uint8)
    new_HSV_img = HSV_img_pixels.transpose(1, 0).reshape((3, height, width)).transpose(1, 2, 0)
    RGB_img_pixels = cv2.cvtColor(new_HSV_img, cv2.COLOR_HSV2RGB)

    # Show and save result
    imshow(RGB_img_pixels, f"Mapped to color axis ({hue_degree} degrees)")
    img_path = results_dir + img_file.replace(".jpg", f"_{hue_degree}degrees_{rescale_shift}rescale_shift.jpg")
    cv2.imwrite(img_path, RGB_img_pixels[:, :, ::-1])

    # Plot new color wheel
    plot_color_wheel(new_HSV_img, results_dir+f"colorwheel_new.jpg", downsampling_factor=1)

    return


def main():
    """Example usage below."""

    # Real image
    img_dir = "images/" # Image directory=
    img_file = "nicholas-roerich_tent-mountain-1933.jpg" # Image file to modify
    results_dir = "results/image_recoloring/examples/" # Where to save the altered images
    os.makedirs(results_dir, exist_ok=True) # Creates results_dir if it does not exist

    """ Here we plot the image pixel distribution on the color wheel."""
    hsv_img, rgb_img = get_HSV_RGB(img_dir + img_file, scale_factor=1)
    plot_color_wheel(hsv_img, results_dir+"colorwheel.jpg", downsampling_factor=10)
    print(hsv_img.shape)

    # Synthetic image
    hsv_img = np.array(([[[0, 240, 255], [90, 240, 255], [140, 240, 255]]])).astype(np.uint8)
    repeat_pixels = 1
    hsv_img = np.repeat((hsv_img), repeat_pixels, axis=0)
    imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB), img_file)
    plot_color_wheel(hsv_img, results_dir + "colorwheel.jpg", downsampling_factor=1)

    """ Image recoloring image by moving the pixels onto axis 30-210 degrees """
    hue_degree = 30
    rescale_shift = 1.5
    recolor_image(hsv_img, hue_degree, rescale_shift, img_file, results_dir)


if __name__ == "__main__":
    main()