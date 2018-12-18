from scipy import misc
import numpy as np


def crop(image_path_list, rect_list, dest_image_size, margin):
    assert len(image_path_list) == len(rect_list)
    cropped_list = []

    for i in range(len(image_path_list)):
        image_path = image_path_list[i]
        rect = rect_list[i]

        left = rect['left']
        right = rect['right']
        top = rect['top']
        bottom = rect['bottom']

        img = misc.imread(image_path)

        img_size = np.asarray(img.shape)[0:2]

        left = int(max(left - margin / 2, 0))
        right = int(min(right + margin / 2, img_size[0]))
        top = int(max(top - margin / 2, 0))
        bottom = int(min(bottom + margin / 2, img_size[1]))

        cropped = img[left:right, top:bottom, :]
        scaled = misc.imresize(cropped, (dest_image_size, dest_image_size), interp='bilinear')

        cropped_list.append(scaled)

    return np.array(cropped_list)


def crop_one(image_path, rect, dest_image_size, margin):
    cropped_list = crop([image_path], [rect], dest_image_size=dest_image_size, margin=margin)

    if cropped_list.shape[0] > 0:
        return cropped_list[0]
    else:
        return None
