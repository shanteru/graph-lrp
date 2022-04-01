# MIT License
#
# Copyright (c) 2020 Hryhorii Chereda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
    Contains two implementation of Chebyshev convolutional layer from the paper [Convolutional Neural Networks
    on Graphs with Fast Localized Spectral
    Filtering](https://arxiv.org/abs/1606.09375)<br>
    Michaël Defferrard et al.
    One implementation of the convolution is the same as in original paper.
    Another one has a slower implementation and memory more demanding, but SHAP's Deep Explained is applicable to it.
"""


from tensorflow import keras
import tensorflow as tf

#from tensorflow.keras import backend as KB
from tensorflow.keras.layers import Layer

import scipy.sparse
import numpy as np

from lib import graph

class ChebConv(Layer):
    """
    Chebyshev convolutional layer from the paper [Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering](https://arxiv.org/abs/1606.09375)<br>
    Michaël Defferrard et al.
    The implementation of the convolution is the same as in original paper.
    Deep Explainer from SHAP is not applicable to the models built with this layer, because it uses
    fast version of sparse to dense multiplications. Deep Explainer from SHAP is applicable to ChebConvSlow layer. The
    latest uses dense*dense matrix that Deep Explainer from SHAP can deal with.

    Input of the layer:
        - Node features of shape ([batch], n_nodes, n_node_features).
    Output of the layer:
        - Node features with the same [batch] and n_nodes, but with the n_node_features
        equal to the number of filters.

    Arguments to construct the layer
        channels: number of output channels.
        L: laplacian matrix (single) of the graph.
        K: order of the Chebyshev polynomials.
        activation: activation function.
        use_bias: bool, add a bias vector to the output.
        kernel_initializer`: initializer for the weights.
        bias_initializer: initializer for the bias vector.
        kernel_regularizer: regularization applied to the weights.
        bias_regularizer: regularization applied to the bias vector.
        activity_regularizer: regularization applied to the output.
        kernel_constraint: constraint applied to the weights.
        bias_constraint: constraint applied to the bias vector.
    """


    def __init__(
        self,
        channels,
        L,
        K=2,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform", # keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(ChebConv, self).__init__(**kwargs)

        self.channels = channels
        self.K = K

        self.L = self.prepare_laplacian(L) #calc_Laplace_Polynom(L, K)

        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        # self.input_spec = InputSpec(ndim=3)

    @staticmethod
    def prepare_laplacian(L):
        """
        Preparing laplacian matrix of the convolutional layer in tensorflow format.
        """
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.sparse.SparseTensor(indices, L.data, L.shape)
        # print("\n\tPrepare Laplacian\n")
        return tf.sparse.reorder(L)

    def build(self, input_shape):
        # print("\t", "input_shape", input_shape) # (size in a batch, feature_num, channel)
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.K * input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1, 1, self.channels),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, input):
        x = input
        Fout = self.channels
        _, M, Fin = x.get_shape()
        N = tf.shape(x)[0] # N, M, Fin
        # print("\t", "x inside call", x.get_shape().as_list())

        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # print("\t", "x0 inside call", x0.get_shape().as_list())
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if self.K > 1:
            x1 = tf.sparse.sparse_dense_matmul(self.L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * tf.sparse.sparse_dense_matmul(self.L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2

        x = tf.reshape(x, [self.K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K

        x = tf.reshape(x, [N * M, Fin * self.K])  # N*M x Fin*K

        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.

        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [N, M, Fout])
        return tf.nn.relu(x + self.bias)

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K}




class ChebConvSlow(Layer):
    """
    Slow version of a Chebyshev convolutional layer from the paper
    [Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering](https://arxiv.org/abs/1606.09375)<br>
    Michaël Defferrard et al.
    Deep Explainer from SHAP is applicable to the models built with this layer. This layer uses dense*dense matrix
    multiplication, while the fast version uses sparse to dense multiplications.

    Input of the layer:
        - Node features of shape ([batch], n_nodes, n_node_features).
    Output of the layer:
        - Node features with the same [batch] and n_nodes, but with the n_node_features
        equal to the number of filters.

    Arguments to construct the layer
        channels: number of output channels.
        L: laplacian matrix (single) of the graph.
        K: order of the Chebyshev polynomials.
        activation: activation function.
        use_bias: bool, add a bias vector to the output.
        kernel_initializer`: initializer for the weights.
        bias_initializer: initializer for the bias vector.
        kernel_regularizer: regularization applied to the weights.
        bias_regularizer: regularization applied to the bias vector.
        activity_regularizer: regularization applied to the output.
        kernel_constraint: constraint applied to the weights.
        bias_constraint: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        L,
        K=2,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(ChebConvSlow, self).__init__(**kwargs)

        self.channels = channels
        self.K = K


        # Main difference that requires plenty of memory
        self.L = self.prepare_laplacian(L) #calc_Laplace_Polynom(L, K)
        self.L = tf.sparse.to_dense(self.L)

        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        # self.input_spec = InputSpec(ndim=3)

    @staticmethod
    def prepare_laplacian(L):
        """
        Preparing laplacian matrix of the convolutional layer in tensorflow format.
        """
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.sparse.SparseTensor(indices, L.data, L.shape)
        # print("\n\tPrepare Laplacian\n")
        return tf.sparse.reorder(L)

    def build(self, input_shape):
        # print("\t", "input_shape", input_shape) # (size in a batch, feature_num, channel)
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.K * input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1, 1, self.channels),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, input):
        x = input
        Fout = self.channels
        _, M, Fin = x.get_shape()
        N = tf.shape(x)[0] # N, M, Fin
        # print("\t", "x inside call", x.get_shape().as_list())

        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # print("\t", "x0 inside call", x0.get_shape().as_list())
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if self.K > 1:
            x1 = tf.matmul(self.L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * tf.matmul(self.L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2

        x = tf.reshape(x, [self.K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K

        x = tf.reshape(x, [N * M, Fin * self.K])  # N*M x Fin*K

        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.

        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [N, M, Fout])
        return tf.nn.relu(x + self.bias)

    def old_call(self, input):
        x = input
        Fout = self.channels
        _, M, Fin = x.get_shape()
        N = tf.shape(x)[0] # N, M, Fin
        # print("\t", "x inside call", x.get_shape().as_list())

        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # print("\t", "x0 inside call", x0.get_shape().as_list())
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if self.K > 1:
            x1 = tf.sparse.sparse_dense_matmul(self.L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * tf.sparse.sparse_dense_matmul(self.L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2

        x = tf.reshape(x, [self.K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K

        x = tf.reshape(x, [N * M, Fin * self.K])  # N*M x Fin*K

        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.

        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [N, M, Fout])
        return tf.nn.relu(x + self.bias)

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K}


class NonPos(keras.constraints.Constraint):
    """Constrains the weights to be non-positive.
    Property needed for biases when applying the framework of deep taylor decomposition.
    """
    def __call__(self, w):
        return w * tf.cast(tf.less_equal(w, 0.), keras.backend.floatx())
