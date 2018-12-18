import numpy as np
import math


def standardize(images):
    image_num, width, height, channel = images.shape

    mean = np.mean(images, axis=(1, 2, 3))
    stddev = np.std(images, axis=(1, 2, 3))

    tmp = 1.0 / math.sqrt(width + height + channel)
    stddev[stddev < tmp] = tmp

    assert (stddev.shape == (image_num,))
    assert (mean.shape == (image_num,))

    mean = mean.reshape((image_num, 1, 1, 1))
    stddev = stddev.reshape((image_num, 1, 1, 1))

    return (images - mean) / stddev


def standardize_one(image):
    stddev = np.std(image).squeeze()

    width, height, channel = image.shape

    tmp = 1.0 / math.sqrt(width + height + channel)

    adjusted_stddev = max(stddev, tmp)

    mean = np.mean(image)

    return (image - mean) / adjusted_stddev
