#!python


"""
The file contains code to create ChebNet as a Keras Sequential model,
as well Keras and Sklearn models. The models can be accessible through the same interface.
"""

random_state = 7
seed_value= random_state

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

from tensorflow.keras import losses, optimizers, regularizers, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling2D, Flatten, AveragePooling1D, MaxPooling1D, Dropout
#from tensorflow.keras.layers.advanced_activations import ELU
from sklearn.ensemble import RandomForestClassifier
from components import cheb_conv



class MyModel:
    """Interface for the application of k-fold cross validation on Keras neural networks and on Scipy Random Forest. """
    def create(self):
        """To create the model."""
        pass
    def fit(self):
        """Fitting the machine learning method.The creation of the model should be also performed here."""
        pass

    def predict(self):
        """Getting the probabilities of classes"""
        pass

class MyNN(MyModel):
    """Creates the interface for NN Keras model that is used for the cross validation."""
    def __init__(self, params):
        self.params = params
        self.model = None

    def create(self, feature_number):
        random.seed(random_state)
        tf.random.set_seed(random_state)
        self.model = get_nn_model(feature_number=feature_number, regulrztn=self.params["regularization"], learning_rate=self.params["learning_rate"], momentum=self.params["momentum"], decay=self.params["decay"])
        # self.model = get_ELU_nn_model(feature_number=feature_number, regulrztn=self.params["regulrztn"], learning_rate=self.params["learning_rate"], momentum=self.params["momentum"], decay=self.params["decay"])
        # self.model = get_logistic_regression_model(feature_number, regulrztn=self.params["regulrztn"], learning_rate=self.params["learning_rate"], momentum=self.params["momentum"], decay=self.params["decay"])

    def fit(self, x, y, validation_data=None, verbose=0):
        C = np.unique(y).shape[0]
        I = np.eye(C)
        y = I[y]
        validation_data[1] = I[validation_data[1]]
        return self.model.fit(x=x, y=y, batch_size=self.params["batch_size"], epochs=self.params["num_epochs"],
                                 validation_data=(validation_data[0], validation_data[1]), verbose=verbose)

    def predict(self, X_test):
        y_preds = np.squeeze(self.model.predict(X_test))
        #print(y_preds)
        return y_preds
        # y_preds = list(y_preds)
        # y_preds = list(map(lambda x: [1 - x, x], y_preds))
        # return np.array(y_preds)

class MyChebNet(MyModel):
    """Creates the same interface of the ChebNet Keras model that is used for the cross validation."""
    def __init__(self, params):
        self.params = params
        self.model = None

    def create(self, feature_number):
        """Feature_number is a fake parameter, just to match the signatures."""
        tf.random.set_seed(random_state)
        self.model = get_cheb_net_model(**self.params)

    def fit(self, x, y, validation_data=None, class_weight=None, verbose=2):
        C = np.unique(y).shape[0]
        I = np.eye(C)
        y = I[y]
        if validation_data is not None:
            validation_data[1] = I[validation_data[1]]
            return self.model.fit(x=x, y=y, batch_size=self.params["batch_size"], class_weight=class_weight, epochs=self.params["num_epochs"],
                                      validation_data=(validation_data[0], validation_data[1]), verbose=verbose)
        else:
            return self.model.fit(x=x, y=y, batch_size=self.params["batch_size"], epochs=self.params["num_epochs"],
                                  class_weight=class_weight, verbose=verbose)

    def predict(self, X_test):
        y_preds = np.squeeze(self.model.predict(X_test))
        # print(y_preds)
        return y_preds


class MyRF(MyModel):
    """Creates the interface for a Random Forest model that is used for the cross validation."""
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.model = None

    def create(self, feature_number):
        """Feature_number is a fake parameter, just to match the signatures."""

        self.model = RandomForestClassifier(n_estimators=self.n_trees)

    def fit(self, x, y, validation_data=None, verbose=0):
        self.model.fit(x, y)

    def predict(self, X_test):
        return self.model.predict_proba(X_test)


class MyLassoLogisticRegression(MyModel):
    def __init__(self, penalty="l1", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver="warn", max_iter=100, multi_class="warn", verbose=0,
                 warm_start=False, n_jobs=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.model = None

    def create(self, feature_number):
        """Feature_number is a fake parameter, just to match the signatures."""
        self.model = LogisticRegression(self.penalty, self.dual, self.tol, self.C, self.fit_intercept,
                                        self.intercept_scaling, self.class_weight, self.random_state, self.solver,
                                        self.max_iter, self.multi_class, self.verbose, self.warm_start, self.n_jobs)

    def fit(self, x, y, validation_data=None, verbose=0):
        self.model.fit(x, y)

    def predict(self, X_test):
        return self.model.predict_proba(X_test)


def get_logistic_regression_model(feature_number, regulrztn=0, learning_rate=1e-4, momentum=0.9, decay=0.9):
    """
    Hard coded logistic regression model without regularization.
    :return: Keras type model
    """
    print("decay,", decay)
    model = Sequential()
    model.add(Dense(units=1, input_dim=feature_number, activation='sigmoid', kernel_regularizer=None,
                    bias_regularizer=None))
    model.compile(loss=losses.binary_crossentropy,
                  # optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False),
                  optimizer=optimizers.Adam(lr=learning_rate, decay=decay),
                  metrics=["acc"])
    return model



def get_nn_model(feature_number, regulrztn=5e-5, learning_rate=1e-4, momentum=0.9, decay=0.9):
    """
    Creates a hard coded usual NN model. FIVE output inputs.
    TODO: add the parameter for output nodes
    :return: Keras type model
    """
    model = Sequential()
    # model.add(Conv1D(filters=4, kernel_size=kernel_size, input_shape=(None, feature_number), # padding="same",
    #                  activation='relu'))
    #                  #kernel_regularizer=regularizers.l2(regulrztn), bias_regularizer=regularizers.l2(regulrztn)))

    # model.add(MaxPooling1D())
    # model.add(Conv1D(filters=4, kernel_size=kernel_size, padding="same", activation='relu',
    #                  kernel_regularizer=regularizers.l2(regulrztn), bias_regularizer=regularizers.l2(regulrztn)))
    # model.add(Conv1D(filters=4, kernel_size=kernel_size, padding="same", activation='relu',
    #                  kernel_regularizer=regularizers.l2(regulrztn), bias_regularizer=regularizers.l2(regulrztn)))
    # model.add(Flatten())
    model.add(Dense(units=1024, input_dim=feature_number, activation="relu", kernel_regularizer=regularizers.l2(regulrztn),
              bias_regularizer=regularizers.l2(regulrztn)))

    #model.add(Dropout(0.2))

    model.add(Dense(units=1024, activation="relu", kernel_regularizer=regularizers.l2(regulrztn),
                    bias_regularizer=regularizers.l2(regulrztn)))

    #model.add(Dropout(0.2))

    model.add(Dense(units=1024, activation="relu", kernel_regularizer=regularizers.l2(regulrztn),
                    bias_regularizer=regularizers.l2(regulrztn)))
    #model.add(Dropout(0.2))

    # model.add(Dense(units=32, activation="relu", kernel_regularizer=regularizers.l2(regulrztn),
    #                 bias_regularizer=regularizers.l2(regulrztn)))
    #model.add(Dropout(0.2))
    # model.add(Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(regulrztn),
    #                 bias_regularizer=regularizers.l2(regulrztn)))

    model.add(Dense(units=5, activation='softmax', kernel_regularizer=regularizers.l2(regulrztn),
                    bias_regularizer=regularizers.l2(regulrztn)))
    # model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(regulrztn),
    #                 bias_regularizer=regularizers.l2(regulrztn)))
    # model.compile(loss=losses.binary_crossentropy,
    #               # optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False),
    #               optimizer=optimizers.Adam(lr=learning_rate, decay=decay),
    #               metrics=["acc"])
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=7,
        decay_rate=decay)
    model.compile(loss=losses.categorical_crossentropy,
                  # optimizer=optimizers.SGD(learning_rate=lr_schedule, momentum=momentum),
                  optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  metrics=["acc"])

    # Fit the model
    # model.fit(X, Y, epochs=150, batch_size=10)
    # # evaluate the model
    # scores = model.evaluate(X, Y)
    return model

def get_cheb_net_model(L, F, K, p, M,
                 learning_rate=0.1, decay_rate=0.95, decay_steps=None, regularization=0,
                 num_epochs=20,
                 dropout=0, batch_size=100, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                 eval_frequency=200, momentum=0.9, dir_name=''):

    """Constructs graph convolutional neural network (ChebNet) as a Keras model. Utilizes the Keras version of the Graph
    convolutional layer, see components/cheb_conv.py
    The implementaion folows the paper (and code) from Michaël Defferrard, Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering](https://arxiv.org/abs/1606.09375)
    This Keras version of the ChebNet is slightly faster in training than its equivalent model in the code
    corresponding to the paper above.
    If you want to use SHAP.DeepExplainer, please use cheb_conv.ChebConvSlow in cheb_conv.ChebConv.
    If using cheb_conv.ChebConvSlow the training runs substantially slower and the usage of the GPU's memory
    is substantially higher.


    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        L: List of Graph Laplacians. Size M x M. One per coarsening level.
        F: Number of graph convolutional filters.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases in fully-connected layers.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size.

    Directories:
        dir_name: Name for directories (summaries and model parameters).

    """

    # Verify the consistency w.r.t. the number of layers.
    assert len(L) >= len(F) == len(K) == len(p)
    assert np.all(np.array(p) >= 1)
    p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
    assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
    assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.

    # Keep the useful Laplacians only. May be zero.
    feature_num = L[0].shape[0]
    j = 0
    L_param = []
    for pp in p:
        L_param.append(L[j])
        j += int(np.log2(pp)) if pp > 1 else 0

    g_model = Sequential()

    # !!!
    # Putting graph pooling and graph_conv_layers into sequential model
    # TODO: Take care that as in an original implementation, the regularization is not applied to the graph convolutional
    # layers, but you can apply it if you wish.

    for i in range(len(p)):
        if i == 0:
            g_model.add(cheb_conv.ChebConv(channels=F[i], L=L_param[i], K=K[i], activation='relu', use_bias=True, input_shape=(feature_num, 1)))
            print("\n\tCheb_conv, first layer, input shape :", L_param[i].shape[0])

        if i > 0: # No input shape for consecutive conv layers
            g_model.add(cheb_conv.ChebConv(channels=F[i], L=L_param[i], K=K[i], activation='relu', use_bias=True))
            print("\tCheb_conv layer, input shape :", L_param[i].shape[0])

        if p[i] > 1:
            if pool == "mpool1":
                g_model.add(MaxPooling1D(pool_size=p[i], strides=None, padding='same'))
            if pool == "apool1":
                g_model.add(AveragePooling1D(pool_size=p[i], strides=None, padding='same'))
            print("\t" + pool + ",", "Pooling layer, size :", p[i])

    g_model.add(Flatten())

    # !!!
    # Putting fully-connected layers into sequential model. Possibly with dropout.
    for i, M_i in enumerate(M[:-1]):
        print("\tFC layer, nodes:", M_i)
        g_model.add(Dense(M_i, activation='relu', kernel_regularizer=regularizers.l2(regularization), bias_regularizer=regularizers.l2(regularization)))
        if dropout != 1:
            g_model.add(Dropout(rate=dropout))
            print("\t\tDropout, rate :", dropout)

    # !!!
    # Adding the last softmax layer.
    g_model.add(Dense(units=M[-1], activation='softmax', kernel_regularizer=regularizers.l2(regularization),
                    bias_regularizer=regularizers.l2(regularization))) # , kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None))
    print("\tLast layer, nodes:", M[-1], "\n")

    if decay_rate != 1:
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True # was true
        )
    else:
        lr_schedule = learning_rate

    if momentum == 0:
        optimizer = optimizers.Adam(learning_rate=lr_schedule)  # changed by Greg
    else:
        optimizer = optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)

    g_model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=["acc"])
    g_model.summary()
    return g_model
