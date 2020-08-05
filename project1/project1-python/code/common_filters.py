#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/5 上午9:55
# @Author  : ChengLu
# @File    : common_filters.py
# @Contact : 2854859592@qq.com
import cv2
import numpy as np

def fspecial(r, c, sigma):
    # MATLAB
    # H = fspecial('Gaussian', [r, c], sigma);
    return np.multiply(cv2.getGaussianKernel(r, sigma), (cv2.getGaussianKernel(c, sigma)).T)

identity_filter = np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
                           dtype="float32")

blur_filter = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), dtype="float32")

large_1d_blur_filter = fspecial(25, 1, 10)

sobel_filter = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="float32")

laplacian_filter = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="float32")

