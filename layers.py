from inits import *

# from tensorflow_probability import distributions as tfd

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def dot_transpose_a(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense) and transpose a."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y, adjoint_a=True)
    else:
        res = tf.matmul(x, y, transpose_a=True)
    return res


def dot_transpose_b(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense) and transpose a."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y, adjoint_b=True)
    else:
        res = tf.matmul(x, y, transpose_b=True)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs, support):
        return tf.tuple([inputs, support])

    def __call__(self, inputs, support):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            (outputs, adj) = self._call(inputs, support)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return [outputs, adj]

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Layer1(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs, support):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']
        # output = tf.layers.batch_normalization(output, training=True)
        return [self.act(output), support]


class DenseFlat(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(DenseFlat, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim * 4, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs, support):
        inputs = tf.reshape(inputs, [-1, 512])

        x = tf.compat.v1.layers.flatten(inputs)

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']
        # output = tf.layers.batch_normalization(output, training=True)
        return [self.act(output), support]


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.regularizers = []

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs, support):
        x = inputs
        self.support = support
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support1 = dot(support[i], pre_sup, sparse=True)
            supports.append(support1)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        output = tf.math.l2_normalize(output, -1)

        return [self.act(output), support]

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var


class GraphDiffPooling(Layer):
    """Graph Differentiable Pooling layer."""

    def __init__(self, input_dim, cluster_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphDiffPooling, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.support = []
        self.act = act
        self.link_pred_loss = 0
        self.entropy_loss = 0
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.S = []
        self.z = []
        self.embed = GraphConvolution(input_dim=input_dim,
                                      output_dim=output_dim,
                                      placeholders=placeholders,
                                      act=lambda x: x,
                                      dropout=False,
                                      logging=self.logging)
        self.pool = GraphConvolution(input_dim=input_dim,
                                     output_dim=cluster_dim,
                                     placeholders=placeholders,
                                     act=lambda x: x,
                                     dropout=False,
                                     logging=self.logging)

    def _call(self, inputs, support):
        x = inputs
        a = support
        self.support = support
        (S, _) = self.pool(x, a)
        (Z, _) = self.embed(x, a)

        S = tf.nn.softmax(S, axis=-1)
        coarse_x = dot_transpose_a(S, Z)
        ss_t = dot_transpose_b(S, S)
        self.S = ss_t
        self.Z = S
        coarse_a = list()
        tmp_pred_loss = list()
        for i in range(len(a)):
            coarse_a_temp = dot(a[i], S, self.sparse_inputs)
            coarse_a_temp = dot_transpose_a(S, coarse_a_temp)
            zero = tf.constant(0, dtype=tf.float32)
            where = tf.not_equal(coarse_a_temp, zero)
            indices = tf.where(where)
            values = tf.gather_nd(coarse_a_temp, indices)
            sparse = tf.SparseTensor(indices, values, coarse_a_temp.shape)
            sparse = tf.sparse_reorder(sparse)
            coarse_a.append(sparse)
            a_dense = tf.sparse.to_dense(a[i], validate_indices=False)
            tmp_pred_loss.append(frobenius_norm_tf(tf.subtract(a_dense, ss_t)))

        self.link_pred_loss = tf.reduce_mean(tmp_pred_loss)
        self.entropy_loss = tf.reduce_mean(entropy(S))
        coarse_x = tf.math.l2_normalize(coarse_x, -1)

        return [coarse_x, coarse_a]


def cross_entropy(x, y, axis=-1):
    safe_y = tf.where(tf.equal(x, 0.), tf.ones_like(y), y)
    return -tf.reduce_sum(x * tf.log(safe_y), axis)


def entropy(x, axis=-1):
    return cross_entropy(x, x, axis)


def frobenius_norm_tf(M):
    # tf.reduce_sum(M ** 2) ** 0.5
    return tf.norm(M, ord='fro', axis=[-2, -1])
