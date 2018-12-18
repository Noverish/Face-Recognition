from tensorflow.python.platform import gfile
import tensorflow as tf
import os

model_path = os.path.join(os.path.split(__file__)[0], '../../models/embedding/pretrained2.pb')


def embed(images):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            with gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=None, name='')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

            return emb_array


def embed_one(image):
    tmp = embed([image])

    if len(tmp) > 0:
        return tmp[0]
    else:
        return None
