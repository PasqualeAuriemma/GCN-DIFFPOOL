from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.adj = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        self.adj.append(self.placeholders['support'])
        for layer in self.layers:
            (hidden, hiddenAdj) = layer(self.activations[-1], self.adj[-1])
            self.activations.append(hidden)
            self.adj.append(hiddenAdj)
        self.outputs = self.activations[-1]
        self.outputs = self.predict()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class Model1(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model1):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


def frobenius_norm_tf(M):
    # tf.reduce_sum(M ** 2) ** 0.5
    return tf.norm(M, ord='fro', axis=[-2, -1])


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss

        for i in [1,3]:  # [2, 5, 8, 11]:
            self.loss += self.layers[i].entropy_loss + self.layers[i].link_pred_loss
        self.loss = FLAGS.weight_decay * tf.nn.l2_loss(self.loss)
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[2].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[4].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #    Weight decay loss
        for var in self.layers[5].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        # for var in self.layers[7].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=self.input_dim,
        #                                     output_dim=self.input_dim,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     bias=True,
        #                                     dropout=True,
        #                                     sparse_inputs=False,
        #                                     logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
                                            cluster_dim=FLAGS.cluster1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act= tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                             output_dim=FLAGS.hidden2,
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             bias=True,
                                             dropout=True,
                                             logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                      output_dim=FLAGS.hidden2,
        #                                      placeholders=self.placeholders,
        #                                      act=tf.nn.relu,
        #                                      bias=True,
        #                                      dropout=True,
        #                                      logging=self.logging))
        # self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
        #                                     cluster_dim=FLAGS.cluster2,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     bias=False,
        #                                     dropout=True,
        #                                     sparse_inputs=True,
        #                                     logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden3,
        #                                      output_dim=FLAGS.hidden3,
        #                                      placeholders=self.placeholders,
        #                                      act=tf.nn.relu,
        #                                      bias=True,
        #                                      dropout=True,
        #                                      logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden3,
        #                                      output_dim=FLAGS.hidden3,
        #                                      placeholders=self.placeholders,
        #                                      act=tf.nn.relu,
        #                                      bias=True,
        #                                      dropout=True,
        #                                      logging=self.logging))
        # self.layers.append(GraphDiffPooling(input_dim=FLAGS.hidden3,
        #                                     cluster_dim=FLAGS.cluster3,
        #                                     output_dim=FLAGS.hidden4,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     bias=False,
        #                                     dropout=True,
        #                                     sparse_inputs=True,
        #                                     logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                      output_dim=FLAGS.hidden1,
        #                                      placeholders=self.placeholders,
        #                                      act=tf.nn.relu,
        #                                      bias=True,
        #                                      dropout=True,
        #                                      logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden4,
        #                                      output_dim=FLAGS.hidden4,
        #                                      placeholders=self.placeholders,
        #                                      act=tf.nn.relu,
        #                                      bias=True,
        #                                      dropout=True,
        #                                      logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=FLAGS.hidden2,
                                            cluster_dim=1,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=False,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # self.layers.append(DenseFlat(input_dim=FLAGS.hidden4,
        #                          output_dim=int(FLAGS.hidden3),
        #                          placeholders=self.placeholders,
        #                          act=lambda x: x,
        #                          bias=True,
        #                          dropout=True,
        #                          logging=self.logging))
        #
        self.layers.append(Dense(input_dim=int(FLAGS.hidden4),
                                 output_dim=FLAGS.hidden3,
                                 placeholders=self.placeholders,
                                 act= tf.nn.relu,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden3),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act= lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_1_strati_1(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_1_strati_1, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for i in [1]:
            self.loss += self.layers[i].entropy_loss + self.layers[i].link_pred_loss
        self.loss = FLAGS.weight_decay * tf.nn.l2_loss(self.loss)
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[2].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[3].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
                                            cluster_dim=1,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=False,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden4),
                                 output_dim=FLAGS.hidden3,
                                 placeholders=self.placeholders,
                                 act= tf.nn.relu,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden3),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act= lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_1_strati_4(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_1_strati_4, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss of pooling layer
        for i in [1]:
            self.loss += self.layers[i].entropy_loss + self.layers[i].link_pred_loss
        self.loss = FLAGS.weight_decay * tf.nn.l2_loss(self.loss)
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[2].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[3].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
                                            cluster_dim=4,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=False,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(DenseFlat(input_dim=FLAGS.hidden4,
                                 output_dim=int(FLAGS.hidden3),
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden3),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act= lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_2_strati_1(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_2_strati_1, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss for pooling layer
        for i in [1, 3]:
            self.loss += self.layers[i].entropy_loss + self.layers[i].link_pred_loss
        self.loss = FLAGS.weight_decay * tf.nn.l2_loss(self.loss)
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[2].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[4].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #    Weight decay loss
        for var in self.layers[5].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
                                            cluster_dim=FLAGS.cluster1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act= tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                             output_dim=FLAGS.hidden2,
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             bias=True,
                                             dropout=True,
                                             logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=FLAGS.hidden2,
                                            cluster_dim=1,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden4),
                                 output_dim=FLAGS.hidden3,
                                 placeholders=self.placeholders,
                                 act= tf.nn.relu,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden3),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act= lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_2_strati_4(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_2_strati_4, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss for pooling layer
        for i in [1, 3]:
            self.loss += self.layers[i].entropy_loss + self.layers[i].link_pred_loss
        self.loss = FLAGS.weight_decay * tf.nn.l2_loss(self.loss)
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[2].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[4].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #    Weight decay loss
        for var in self.layers[5].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
                                            cluster_dim=FLAGS.cluster1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act= tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                             output_dim=FLAGS.hidden2,
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             bias=True,
                                             dropout=True,
                                             logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=FLAGS.hidden2,
                                            cluster_dim=4,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=False,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(DenseFlat(input_dim=FLAGS.hidden4,
                                 output_dim=int(FLAGS.hidden3),
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden3),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act= lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_2_strati_1_2conv(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_2_strati_1_2conv, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for i in [2,5]:
            self.loss += self.layers[i].entropy_loss + self.layers[i].link_pred_loss
        self.loss = FLAGS.weight_decay * tf.nn.l2_loss(self.loss)
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[3].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[4].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[6].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[7].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x:x,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
                                            cluster_dim=FLAGS.cluster1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act= tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                             output_dim=FLAGS.hidden2,
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             bias=True,
                                             dropout=True,
                                             logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x:x,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=FLAGS.hidden2,
                                            cluster_dim=1,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=False,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden4),
                                 output_dim=FLAGS.hidden3,
                                 placeholders=self.placeholders,
                                 act= tf.nn.relu,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden3),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act= lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_1_strati_1_2conv(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_1_strati_1_2conv, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for i in [2]:
            self.loss += self.layers[i].entropy_loss + self.layers[i].link_pred_loss
        self.loss = FLAGS.weight_decay * tf.nn.l2_loss(self.loss)
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        for var in self.layers[3].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # Weight decay loss
        for var in self.layers[4].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x:x,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))
        self.layers.append(GraphDiffPooling(input_dim=self.input_dim,
                                            cluster_dim=1,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=False,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden4),
                                 output_dim=FLAGS.hidden3,
                                 placeholders=self.placeholders,
                                 act= tf.nn.relu,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=int(FLAGS.hidden3),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act= lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

