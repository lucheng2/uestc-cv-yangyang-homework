import time

import cv2
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    # print('my_imfilter function in student.py needs to be implemented')

    # 限制kernel的shape
    if len(kernel.shape) != 2:
        raise TypeError("kernel must be 2D!")
    if (kernel.shape[0] % 2 == 0) or (kernel.shape[1] % 2 == 0):
        raise ValueError("Only support odd-dimension filters!")

    if len(image.shape) == 2:
        # 这部分是库函数返回的结果
        # temp = cv2.filter2D(image, -1, kernel)
        # filtered_image = temp

        filter_width = int(kernel.shape[0] / 2)
        filter_height = int(kernel.shape[1] / 2)

        image_width = image.shape[0]
        image_height = image.shape[1]

        width_padding = np.zeros([image_width, filter_height], dtype=np.uint8) * 255
        height_padding = np.zeros([filter_width, image_height + filter_height * 2], dtype=np.uint8) * 255

        output = np.ones_like(image)
        time_start = time.time()
        print("Filtering...")
        for channel in range(0, 3):
            # 对每个通道进行计算
            channel_data = image[:, :, channel]
            # 上下的padding
            channel_data = np.concatenate([width_padding, channel_data], axis=1)
            channel_data = np.concatenate([channel_data, width_padding], axis=1)
            # 左右的padding
            channel_data = np.concatenate([height_padding, channel_data], axis=0)
            channel_data = np.concatenate([channel_data, height_padding], axis=0)

            # 让filter划过整个通道的长宽
            for column in range(filter_width, channel_data.shape[0] - filter_width):
                for row in range(filter_height, channel_data.shape[1] - filter_height):
                    ret = np.multiply(kernel, channel_data[column - filter_width:column + filter_width + 1,
                                              row - filter_height:row + filter_height + 1])
                    # 保存对应位
                    output[column - filter_width, row - filter_height, channel] = min(max(int(np.sum(ret)), 0), 255)

        time_end = time.time()
        print("End, total:{}".format(time_end - time_start))
        filtered_image = output

    elif len(image.shape) == 3:
        # 这部分是库函数返回的结果
        # temp = cv2.filter2D(image, -1, kernel)
        # filtered_image = temp

        filter_width = int(kernel.shape[0] / 2)
        filter_height = int(kernel.shape[1] / 2)

        image_width = image.shape[0]
        image_height = image.shape[1]

        width_padding = np.zeros([image_width, filter_height], dtype=np.uint8) * 255
        height_padding = np.zeros([filter_width, image_height + filter_height * 2], dtype=np.uint8) * 255

        output = np.ones_like(image)
        time_start = time.time()
        print("Filtering...")
        for channel in range(0, 3):
            # 对每个通道进行计算
            channel_data = image[:, :, channel]
            # 上下的padding
            channel_data = np.concatenate([width_padding, channel_data], axis=1)
            channel_data = np.concatenate([channel_data, width_padding], axis=1)
            # 左右的padding
            channel_data = np.concatenate([height_padding, channel_data], axis=0)
            channel_data = np.concatenate([channel_data, height_padding], axis=0)

            # 让filter划过整个通道的长宽
            for column in range(filter_width, channel_data.shape[0] - filter_width):
                for row in range(filter_height, channel_data.shape[1] - filter_height):
                    ret = np.multiply(kernel, channel_data[column - filter_width:column + filter_width + 1,
                                              row - filter_height:row + filter_height + 1])
                    # 保存对应位
                    temp = np.sum(ret)
                    temp = min(max(int(np.sum(ret)), 0), 255)
                    output[column - filter_width, row - filter_height, channel] = min(max(int(np.sum(ret)), 0), 255)

        time_end = time.time()
        print("End, total:{}".format(time_end - time_start))
        filtered_image = output

    else:
        # 非images
        raise ValueError("unsopport image scale!")
    ##################

    return filtered_image


def conv(image, filter, image_center_x, image_center_y):
    """
    卷积操作
    :param image:
    :param filter:
    :param image_center_x:
    :param image_center_y:
    :return:
    """
    np.convolve()
    size = 3
    radius = int((size - 1) / 2)
    view = np.zeros((size, size, 3), dtype=np.float)
    for i in range(size):
        for j in range(size):
            for z in range(3):
                view[i][j][z] = image[image_center_x - radius + i][image_center_y - radius + j][z] * filter[i][j][z]
    return np.sum(view)


"""
EXTRA CREDIT placeholder function
"""


def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the project webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency * 2
    probs = np.asarray([exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here:
    low_frequencies = np.zeros(image1.shape)  # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = np.zeros(image1.shape)  # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = np.zeros(image1.shape)  # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.

    return low_frequencies, high_frequencies, hybrid_image
