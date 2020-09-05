import os

import matplotlib.pyplot as plt
import numpy as np

from helpers import vis_hybrid_image, load_image, save_image
from student import gen_hybrid_image


# Before trying to construct hybrid images, it is suggested that you
# implement my_imfilter in helpers.py and then debug it using proj1_part1.py

# Debugging tip: You can split your python code and print in between
# to check if the current states of variables are expected.

# Setup
# Read images and convert to floating point format
def main(image_name_1, image_name_2):
    image_1 = os.path.join("../data", image_name_1+".bmp")
    image_2 = os.path.join("../data", image_name_2+".bmp")
    image1 = load_image(image_1)
    image2 = load_image(image_2)

    # display the dog and cat images
    plt.figure(figsize=(3, 3))
    plt.imshow((image1 * 255).astype(np.uint8))
    plt.figure(figsize=(3, 3))
    plt.imshow((image2 * 255).astype(np.uint8))

    # There are several additional test cases in 'data'.
    # Feel free to make your own, too (you'll need to align the images in a
    # photo editor such as Photoshop).
    # The hybrid images will differ depending on which image you
    # assign as image1 (which will provide the low frequencies) and which image
    # you asign as image2 (which will provide the high frequencies)

    ## Hybrid Image Construction ##
    # cutoff_frequency is the standard deviation, in pixels, of the Gaussian#
    # blur that will remove high frequencies. You may tune this per image pair
    # to achieve better results.
    cutoff_frequency = 7
    low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(
        image1, image2, cutoff_frequency)

    ## Visualize and save outputs ##
    plt.figure()
    plt.imshow((low_frequencies * 255).astype(np.uint8))
    plt.figure()
    plt.imshow(((high_frequencies + 0.5) * 255).astype(np.uint8))
    vis = vis_hybrid_image(hybrid_image)
    plt.figure(figsize=(20, 20))
    plt.imshow(vis)

    save_image('../results/' + image_name_1 + '_low_frequencies.jpg', low_frequencies)
    outHigh = np.clip(high_frequencies + 0.5, 0.0, 1.0)
    save_image('../results/' + image_name_2 + '_high_frequencies.jpg', outHigh)
    save_image('../results/' + image_name_1 + image_name_2 + '_hybrid_image.jpg', hybrid_image)
    save_image('../results/' + image_name_1 + image_name_2 + '_hybrid_image_scales.jpg', vis)


if __name__ == '__main__':
    image_name_list = [
        ["dog", "cat"],
        ["bicycle", "motorcycle"],
        ["bird", "plane"],
        ["fish", "submarine"],
        ["einstein", "marilyn"]
    ]
    for image_name in image_name_list:
        main(image_name[0], image_name[1])
