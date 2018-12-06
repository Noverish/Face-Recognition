import tensorflow as tf
import cv2
import numpy as np

from src.detection.mtcnn import PNet, RNet, ONet
from src.detection.tools import detect_face, get_model_filenames

minsize = 20
threshold = [0.8, 0.8, 0.8]
factor = 0.7

model_path = './src/detection/mtcnn_model'


def detect(image_path):
    img = cv2.imread(image_path)
    file_paths = get_model_filenames(model_path)
    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                saver = tf.train.import_meta_graph(file_paths[0])
                saver.restore(sess, file_paths[1])

                def pnet_fun(img):
                    return sess.run(
                        ('softmax/Reshape_1:0',
                         'pnet/conv4-2/BiasAdd:0'),
                        feed_dict={
                            'Placeholder:0': img})

                def rnet_fun(img):
                    return sess.run(
                        ('softmax_1/softmax:0',
                         'rnet/conv5-2/rnet/conv5-2:0'),
                        feed_dict={
                            'Placeholder_1:0': img})

                def onet_fun(img):
                    return sess.run(
                        ('softmax_2/softmax:0',
                         'onet/conv6-2/onet/conv6-2:0',
                         'onet/conv6-3/onet/conv6-3:0'),
                        feed_dict={
                            'Placeholder_2:0': img})

                rectangles, points = detect_face(img, minsize,
                                                 pnet_fun, rnet_fun, onet_fun,
                                                 threshold, factor)

                points = np.transpose(points)
                for rectangle in rectangles:
                    cv2.putText(img, str(rectangle[4]),
                                (int(rectangle[0]), int(rectangle[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0))
                    cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])),
                                  (int(rectangle[2]), int(rectangle[3])),
                                  (255, 0, 0), 1)
                for point in points:
                    for i in range(0, 10, 2):
                        cv2.circle(img, (int(point[i]), int(
                            point[i + 1])), 2, (0, 255, 0))
                cv2.imwrite('result.jpg', img)

                return rectangles, points
