from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import src.detection.detect_face
import os
from pprint import pprint

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
gpu_memory_fraction = 1.0

pnet = None
rnet = None
onet = None

model_path = os.path.join(os.path.split(__file__)[0], '../../models/detection/')


def load_mtcnn_net():
    global pnet
    global rnet
    global onet

    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)


def detect(image_path_list, detect_multiple=False, verbose=True):
    global pnet
    global rnet
    global onet

    if pnet is None or rnet is None or onet is None:
        load_mtcnn_net()

    faces = []

    for i in range(len(image_path_list)):
        image_path = image_path_list[i]
        img = misc.imread(image_path)
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        face_num_in_picture = bounding_boxes.shape[0]

        if face_num_in_picture == 0:
            faces.append(None)
            continue

        faces_in_picture = []
        faces_size = []

        for j in range(face_num_in_picture):
            box = bounding_boxes[j]
            rect = {
                'left': box[1],
                'right': box[3],
                'top': box[0],
                'bottom': box[2]
            }
            size = (rect['right'] - rect['left']) * (rect['bottom'] - rect['top'])

            faces_in_picture.append(rect)
            faces_size.append(size)

        if detect_multiple:
            faces.append(faces_in_picture)
        else:
            largest_index = faces_size.index(max(faces_size))
            largest_rect = faces_in_picture[largest_index]
            faces.append(largest_rect)

        if (i + 1) % 100 == 0 and verbose:
            print('Detecting faces... ({}/{})'.format(i + 1, len(image_path_list)))

    return faces


def detect_one(image_path, detect_multiple=False):
    detected_list = detect([image_path], detect_multiple)

    if len(detected_list) > 0:
        return detected_list[0]
    else:
        return None
