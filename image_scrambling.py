"""We follow Albers' method for image shuffling, consisting of shuffling patches + transformation with Eidolon factory:

"For spatial scrambling (S-images), the images were divided in smaller sections of 60*60 pixels, as schematically
indicated with the yellow lines (Fig. 1D), to preserve the local information. The smaller sections were spatially
shuffled and subsequently the images were modulated with the “Eidolon factory”
(Koenderink, Valsecchi, van Doorn, Wagemans, & Gegenfurtner, 2017)."

Anke Marit Albers, Karl R. Gegenfurtner, Sérgio M.C. Nascimento,
An independent contribution of colour to the aesthetic preference for paintings,
Vision Research, Volume 177, 2020, Pages 109-117, ISSN 0042-6989,

This code is based on the official implementation of the Eidolon Factory by Jan Koenderink, available at
https://github.com/gestaltrevision/Eidolon
"""

import os

from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.transforms import Resize

from eidolonfactory.helpers import ImageToOpponentRepresentation, DataPlaneDisarray, OpponentRepresentationToImage
from eidolonfactory.noise import BlurredRandomGaussianDataPlane
from eidolonfactory.picture import Picture


def CoherentDisarrayOfEdgesOpponentChannels(pic, theReach, theGrain):
    print("Embarking on coherent disarray of edges opponent channels")

    (h, w) = pic.fatFiducialDataPlane.shape
    numScaleLevels = pic.numScaleLevels
    scaleLevels = pic.scaleLevels

    print("Computing eidolon...")

    # // B & C COMBINED AND REPEATED THREE TIMES
    pic.color = 'red'  # have to set a color to make the pic produce color data planes
    # need to cast to float otherwise calculations go horribly wrong!
    r = pic.colorFatFiducialDataPlane['red'].astype('float')  # since type = uint8 => cast to type float
    g = pic.colorFatFiducialDataPlane['green'].astype('float')  # since type = uint8 => cast to type float
    b = pic.colorFatFiducialDataPlane['blue'].astype('float')  # since type = uint8 => cast to type float
    kw, rg, yb = ImageToOpponentRepresentation(r, g, b)
    pic.color = None  # reset color, just to be on the safe side

    print("BLACK-WHITE disarray going on...")
    xDisplacements = BlurredRandomGaussianDataPlane(w, h, theGrain).blurredRandomGaussianDataPlane
    yDisplacements = BlurredRandomGaussianDataPlane(w, h, theGrain).blurredRandomGaussianDataPlane
    kwPlane = DataPlaneDisarray(kw, xDisplacements, yDisplacements, theReach)
    print("RED_GREEN disarray going on...")
    xDisplacements = BlurredRandomGaussianDataPlane(w, h, theGrain).blurredRandomGaussianDataPlane
    yDisplacements = BlurredRandomGaussianDataPlane(w, h, theGrain).blurredRandomGaussianDataPlane
    rgPlane = DataPlaneDisarray(rg, xDisplacements, yDisplacements, theReach)
    print("YELLOW_BLUE disarray going on...")
    xDisplacements = BlurredRandomGaussianDataPlane(w, h, theGrain).blurredRandomGaussianDataPlane
    yDisplacements = BlurredRandomGaussianDataPlane(w, h, theGrain).blurredRandomGaussianDataPlane
    ybPlane = DataPlaneDisarray(yb, xDisplacements, yDisplacements, theReach)

    print("Opponent representation to RGB-image...")
    r, g, b = OpponentRepresentationToImage(kwPlane, rgPlane, ybPlane)

    eidolon = np.zeros((pic.embeddingData['h'], pic.embeddingData['w'], 3)).astype('uint8')

    eidolon[:, :, 0] = pic.DisembedDataPlane(r)
    eidolon[:, :, 1] = pic.DisembedDataPlane(g)
    eidolon[:, :, 2] = pic.DisembedDataPlane(b)

    return Image.fromarray(eidolon, 'RGB')


def shuffle_image_patches(img, shuffled_img_path, n_patches=100):
    w, h = img.size

    # Resize images for compatibility with patch size
    n = int(np.sqrt(n_patches))
    resize = Resize((h * n, w * n))
    img_resize = resize(img)
    x = pil_to_tensor(img_resize)

    # Split image in patches and flatten grid
    x = x.view(3, n, h, n, w)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(3, -1, h, w)

    # Shuffle patches
    idx = torch.randperm(x.size(1))
    x = x[:, idx, :, :]

    # Reshape to image grid
    x = x.reshape(3, n, n, h, w)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(3, h * n, w * n)

    # Reshape back to original image size
    img = to_pil_image(x)
    resize = Resize((h, w))
    img = resize(img)

    # Save shuffled image
    img.save(shuffled_img_path)


def main():

    # Define image path and results directory
    img_path = "images/"
    img_name = 'adnan-coker_unknown-title(3).jpg'
    save_dir = "results/image_scrambling/"
    os.makedirs(save_dir, exist_ok=True)

    # Read image and check image size
    img = Image.open(img_path + img_name)
    w, h = img.size
    aspectRatio = float(h) / w

    # Shuffle patches
    n_patches = 100
    shuffled_img_path = f"{save_dir}{img_name.replace('.jpg', '')}_{n_patches}_shuffled_patches.jpg"
    shuffle_image_patches(img, shuffled_img_path, n_patches)

    # Image parameters to set for the Eidolons
    SZ = w if aspectRatio <= 1 else h
    MIN_SIGMA = 1 / np.sqrt(2)
    MAX_SIGMA = SZ / 4.0
    SIGMA_FACTOR = np.sqrt(2)

    # Parameters for the Eidolon factory
    REACH = 22.0
    GRAIN = 8.0

    # Apply Eidolon factory (Coherent Disarray Of Edges) and save the result
    pic = Picture(shuffled_img_path, SZ, MIN_SIGMA, MAX_SIGMA, SIGMA_FACTOR)
    im = CoherentDisarrayOfEdgesOpponentChannels(pic, REACH, GRAIN)
    im.save(save_dir + f"{img_name.replace('.jpg', '_' + str(int(REACH)) + '_' + str(int(GRAIN)) + '.jpg')}")


if __name__ == "__main__":
    main()