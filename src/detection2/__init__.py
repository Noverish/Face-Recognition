from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import src.detection2.detect_face
import os

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
gpu_memory_fraction = 1.0


def detect(image_paths):
    faces = []

    model_path = os.path.join(os.path.split(__file__)[0], 'model')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)

    for image_path in image_paths:
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            print('{}: {}'.format(image_path, e))
            continue

        if img.ndim != 3:
            print('Unable read {}'.format(image_path))
            continue

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        for i in range(bounding_boxes.shape[0]):
            box = bounding_boxes[i]

            faces.append({
                'left': box[1],
                'right': box[3],
                'top': box[0],
                'bottom': box[2]
            })

    return faces
