import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors, BallTree


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    # xs = np.zeros(1)
    # ys = np.zeros(1)

    alpha = 0.04
    ix = ndimage.sobel(image, 0)
    iy = ndimage.sobel(image, 1)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy
    ix2 = ndimage.gaussian_filter(ix2, sigma=2)
    iy2 = ndimage.gaussian_filter(iy2, sigma=2)
    ixy = ndimage.gaussian_filter(ixy, sigma=2)
    c, l = image.shape
    result = np.zeros((c, l))
    har = np.zeros((c, l))
    har_max = 0

    print('looking for corner . . .')
    for i in range(c):
        for j in range(l):
            # print('test ', j)
            m = np.array([[ix2[i, j], ixy[i, j]], [ixy[i, j], iy2[i, j]]], dtype=np.float64)
            har[i, j] = np.linalg.det(m) - alpha * (np.power(np.trace(m), 2))
            if har[i, j] > har_max:
                har_max = har[i, j]

    threshold = 0.01

    print('threshold')
    for i in range(c - 1):
        for j in range(l - 1):
            if har[i, j] > threshold * har_max and har[i, j] > har[i - 1, j - 1] and har[i, j] > har[i - 1, j + 1] \
                    and har[i, j] > har[i + 1, j - 1] and har[i, j] > har[i + 1, j + 1]:
                result[i, j] = 1

    result = np.transpose(result)
    x, y = np.where(result == 1)
    # confidence = [0] * har_max
    return x, y


def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # This is a placeholder - replace this with your features!
    # features = np.zeros((1,128))
    dx = ndimage.sobel(image, 0)
    dy = ndimage.sobel(image, 1)

    features = np.zeros([len(x), 4, 4, 8])

    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx) + math.pi
    angle = np.mod(np.floor(angle / (2 * math.pi) * 8), 8)
    angle = angle.astype(int)

    half_width = feature_width / 2

    for i in range(len(x)):
        px = x[i]
        py = y[i]

        x1 = max(px - half_width, 0)
        x2 = min(px + half_width - 1, image.shape[0])
        y1 = max(py - half_width, 0)
        y2 = min(py + half_width - 1, image.shape[1])

        for row in range(int(x1), int(x2)):
            for col in range(int(y1), int(y2)):
                if col >= 768:
                    print()
                cell_row = np.mod(math.floor((row - x1) / (feature_width / 4)), 4)
                cell_col = np.mod(math.floor((col - y1) / (feature_width / 4)), 4)
                features[i, cell_col, cell_row, angle[col, row]] = \
                    features[i, cell_col, cell_row, angle[col, row]] \
                    + magnitude[col, row]

    features = np.resize(features, [features.shape[0], 128])
    for i in range(features.shape[0]):
        t = max(np.amax(np.sqrt(np.multiply(features[i], features[i]))), 1)
        features[i] = features[i] / t
    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!

    # matches = np.zeros((1,2))
    # confidences = np.zeros(1)

    matches = []
    confidences = []
    ball_tree = BallTree(im1_features, leaf_size=3)

    dist, ind = ball_tree.query(im2_features, 2)

    for i in range(len(ind)):
        index = ind[i]
        distances = dist[i]

        if distances[0] / distances[1] < 0.92:
            matches.append([index[0], i])
            confidences.append(1 - distances[0])
    matches = np.array(matches)
    confidences = np.array(confidences)
    return matches, confidences

