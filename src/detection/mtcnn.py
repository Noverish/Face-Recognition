import tensorflow as tf
import numpy as np


def layer(op):
    def layer_decorated(self, *args, **kwargs):

        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        layer_output = op(self, layer_input, *args, **kwargs)
        tf.add_to_collection('feature_map', layer_output)
        self.layers[name] = layer_output
        self.feed(layer_output)
        return self

    return layer_decorated


class NetWork(object):

    def __init__(self, inputs, trainable=True,
                 weight_decay_coeff=4e-3, mode='train'):

        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.mode = mode
        self.out_put = []
        self.weight_decay_coeff = weight_decay_coeff

        if self.mode == 'train':
            self.tasks = [inp[0] for inp in inputs]
            self.weight_decay = {}
            self.setup_training_graph()
        else:
            self.setup()

    def setup_training_graph(self):

        for index, task in enumerate(self.tasks):
            self.weight_decay[task] = []
            reuse_bool = False
            if index is not 0:
                reuse_bool = True
            self.setup(task=task, reuse=reuse_bool)

    def setup(self, task='data'):

        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, prefix, ignore_missing=False):

        data_dict = np.load(data_path, encoding='latin1').item()
        for op_name in data_dict:
            with tf.variable_scope(prefix + op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):

        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):

        return self.terminals[-1]

    def get_all_output(self):

        return self.out_put

    def get_weight_decay(self):

        assert self.mode == 'train'
        return self.weight_decay

    def get_unique_name(self, prefix):

        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):

        return tf.get_variable(
            name,
            shape,
            trainable=self.trainable,
            initializer=tf.truncated_normal_initializer(
                stddev=1e-4))

    def validate_padding(self, padding):

        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, inp, k_h, k_w, c_o, s_h, s_w, name,
             task=None, relu=True, padding='SAME',
             group=1, biased=True, wd=None):

        self.validate_padding(padding)
        c_i = int(inp.get_shape()[-1])
        assert c_i % group == 0
        assert c_o % group == 0

        def convolve(i, k):
            return tf.nn.conv2d(
                i, k, [1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            kernel = self.make_var(
                'weights', shape=[
                    k_h, k_w, c_i / group, c_o])
            if group == 1:
                output = convolve(inp, kernel)
            else:
                input_groups = tf.split(inp, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i, k in
                                 zip(input_groups, kernel_groups)]
                output = tf.concat(output_groups, 3)
            if (wd is not None) and (self.mode == 'train'):
                self.weight_decay[task].append(
                    tf.multiply(tf.nn.l2_loss(kernel), wd))
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):

        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            return tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name,
                 padding='SAME'):

        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, task=None, relu=True, wd=None):

        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            if (wd is not None) and (self.mode == 'train'):
                self.weight_decay[task] \
                    .append(tf.multiply(tf.nn.l2_loss(weights), wd))
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            return op(feed_in, weights, biases, name=name)

    @layer
    def softmax(self, target, name=None):

        with tf.variable_scope(name):
            return tf.nn.softmax(target, name=name)


class PNet(NetWork):

    def setup(self, task='data', reuse=False):

        with tf.variable_scope('pnet', reuse=reuse):
            (
                self.feed(task).conv(
                    3,
                    3,
                    10,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv1').prelu(
                    name='PReLU1').max_pool(
                    2,
                    2,
                    2,
                    2,
                    name='pool1').conv(
                    3,
                    3,
                    16,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv2').prelu(
                    name='PReLU2').conv(
                    3,
                    3,
                    32,
                    1,
                    1,
                    task=task,
                    padding='VALID',
                    relu=False,
                    name='conv3',
                    wd=self.weight_decay_coeff).prelu(
                    name='PReLU3'))

        if self.mode == 'train':
            if task == 'cls':
                (self.feed('PReLU3')
                 .conv(1, 1, 2, 1, 1, task=task, relu=False,
                       name='pnet/conv4-1', wd=self.weight_decay_coeff))
            elif task == 'bbx':
                (self.feed('PReLU3')
                 .conv(1, 1, 4, 1, 1, task=task, relu=False,
                       name='pnet/conv4-2', wd=self.weight_decay_coeff))
            elif task == 'pts':
                (self.feed('PReLU3')
                 .conv(1, 1, 10, 1, 1, task=task, relu=False,
                       name='pnet/conv4-3', wd=self.weight_decay_coeff))
            self.out_put.append(self.get_output())
        else:
            (self.feed('PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='pnet/conv4-1')
             .softmax(name='softmax'))
            self.out_put.append(self.get_output())
            (self.feed('PReLU3')
             .conv(1, 1, 4, 1, 1, relu=False, name='pnet/conv4-2'))
            self.out_put.append(self.get_output())


class RNet(NetWork):

    def setup(self, task='data', reuse=False):

        with tf.variable_scope('rnet', reuse=reuse):
            (
                self.feed(task).conv(
                    3,
                    3,
                    28,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv1').prelu(
                    name='prelu1').max_pool(
                    3,
                    3,
                    2,
                    2,
                    name='pool1').conv(
                    3,
                    3,
                    48,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv2').prelu(
                    name='prelu2').max_pool(
                    3,
                    3,
                    2,
                    2,
                    padding='VALID',
                    name='pool2').conv(
                    2,
                    2,
                    64,
                    1,
                    1,
                    padding='VALID',
                    task=task,
                    relu=False,
                    name='conv3',
                    wd=self.weight_decay_coeff).prelu(
                    name='prelu3').fc(
                    128,
                    task=task,
                    relu=False,
                    name='conv4',
                    wd=self.weight_decay_coeff).prelu(
                    name='prelu4'))

        if self.mode == 'train':
            if task == 'cls':
                (self.feed('prelu4')
                 .fc(2, task=task, relu=False,
                     name='rnet/conv5-1', wd=self.weight_decay_coeff))
            elif task == 'bbx':
                (self.feed('prelu4')
                 .fc(4, task=task, relu=False,
                     name='rnet/conv5-2', wd=self.weight_decay_coeff))
            elif task == 'pts':
                (self.feed('prelu4')
                 .fc(10, task=task, relu=False,
                     name='rnet/conv5-3', wd=self.weight_decay_coeff))
            self.out_put.append(self.get_output())
        else:
            (self.feed('prelu4')
             .fc(2, relu=False, name='rnet/conv5-1')
             .softmax(name='softmax'))
            self.out_put.append(self.get_output())
            (self.feed('prelu4')
             .fc(4, relu=False, name='rnet/conv5-2'))
            self.out_put.append(self.get_output())


class ONet(NetWork):

    def setup(self, task='data', reuse=False):

        with tf.variable_scope('onet', reuse=reuse):
            (
                self.feed(task).conv(
                    3,
                    3,
                    32,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv1').prelu(
                    name='prelu1').max_pool(
                    3,
                    3,
                    2,
                    2,
                    name='pool1').conv(
                    3,
                    3,
                    64,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv2').prelu(
                    name='prelu2').max_pool(
                    3,
                    3,
                    2,
                    2,
                    padding='VALID',
                    name='pool2').conv(
                    3,
                    3,
                    64,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv3').prelu(
                    name='prelu3').max_pool(
                    2,
                    2,
                    2,
                    2,
                    name='pool3').conv(
                    2,
                    2,
                    128,
                    1,
                    1,
                    padding='VALID',
                    relu=False,
                    name='conv4').prelu(
                    name='prelu4').fc(
                    256,
                    relu=False,
                    name='conv5').prelu(
                    name='prelu5'))

        if self.mode == 'train':
            if task == 'cls':
                (self.feed('prelu5')
                 .fc(2, task=task, relu=False,
                     name='onet/conv6-1', wd=self.weight_decay_coeff))
            elif task == 'bbx':
                (self.feed('prelu5')
                 .fc(4, task=task, relu=False,
                     name='onet/conv6-2', wd=self.weight_decay_coeff))
            elif task == 'pts':
                (self.feed('prelu5')
                 .fc(10, task=task, relu=False,
                     name='onet/conv6-3', wd=self.weight_decay_coeff))
            self.out_put.append(self.get_output())
        else:
            (self.feed('prelu5')
             .fc(2, relu=False, name='onet/conv6-1')
             .softmax(name='softmax'))
            self.out_put.append(self.get_output())
            (self.feed('prelu5')
             .fc(4, relu=False, name='onet/conv6-2'))
            self.out_put.append(self.get_output())
            (self.feed('prelu5')
             .fc(10, relu=False, name='onet/conv6-3'))
            self.out_put.append(self.get_output())
