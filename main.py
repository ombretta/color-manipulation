import os

from HSV_manipulations import get_HSV_RGB, rotate_hue, change_saturation



def main():
    img_dir = "/Users/ombretta/Documents/Code/ansodataset/datasetlocal/"
    img_file = "friedel-dzubas_flowering-2-1963.jpg"
    img_file = "adnan-coker_unknown-title(3).jpg"
    examples_dir = "results/image_recoloring/examples/"
    os.makedirs(examples_dir, exist_ok=True)

    HSV, _ = get_HSV_RGB(img_dir + img_file, scale_factor=0.1)

    degrees = 30
    rotate_hue(HSV.copy(), degrees, 10, img_file, examples_dir)

    saturation_delta = -40
    change_saturation(HSV.copy(), saturation_delta, 10, img_file, examples_dir)


if __name__ == "__main__":
    main()
