#!python

"""
Running the GLRP on GCNN model trained on MNIST data. Digits are graph signals on 8 nearest-neighbors graph.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.datasets import mnist

from components import glrp_scipy, visualize_mnist
from lib import models, graph, coarsening

import time

COARSENING_LEVELS = 4  # to satisfy pooling of size 4 two times we need 4 level
DIR_DATA = "./data/mnist"
METRIC = 'euclidean'
NUMBER_EDGES = 8
M = 28  # size of the digit's picture side
FEATURE_NUM = M * M
EPS = 1e-7  # for adjacency matrix


if __name__ == "__main__":
    # !!!
    # creating the adjacency matrix with all the non-zero weights equal to 1
    z = graph.grid(M)
    dist, idx = graph.distance_sklearn_metrics(z, k=NUMBER_EDGES, metric=METRIC)
    A = graph.adjacency(dist, idx)

    A[A > EPS] = 1

    graphs, perm = coarsening.coarsen(A, levels=COARSENING_LEVELS, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = train_data.reshape((train_data.shape[0], M*M)) / 255
    test_data = test_data.reshape((test_data.shape[0], M*M)) / 255

    print("max, min", np.max(test_data), np.min(test_data))

    train_data = coarsening.perm_data(train_data, perm)
    test_data = coarsening.perm_data(test_data, perm)

    common = {}
    common['dir_name'] = 'mnist_grid_ones/'
    common['num_epochs'] = 30
    common['batch_size'] = 100
    common['decay_steps'] = train_data.shape[0] / common['batch_size']
    common['eval_frequency'] = 2 * train_data.shape[0]/common['batch_size']
    common['brelu'] = 'b1relu'
    common['pool'] = 'mpool1'
    C = max(train_labels) + 1  # number of classes

    common['regularization'] = 5e-4
    common['dropout'] = 0.5
    common['learning_rate'] = 0.03
    common['decay_rate'] = 0.95
    common['momentum'] = 0.9
    common['F'] = [32, 64]
    common['K'] = [25, 25]
    common['p'] = [4, 4]
    common['M'] = [512, C]

    name = 'cgconv_cgconv_softmax_momentum'
    params = common.copy()
    params['dir_name'] += name
    params['filter'] = 'chebyshev5'

    model = models.cgcnn(L, **params)

    # !!!
    # Training
    # In case the trained model is saved: simply comment the three lines below to run glrp again.
    start = time.time()
    accuracy, loss, t_step, trained_losses = model.fit(train_data, train_labels, test_data, test_labels)
    end = time.time()

    probas_ = model.get_probabilities(test_data)
    f1 = 100 * f1_score(test_labels, np.argmax(probas_, axis=1), average='weighted')
    acc = 100 * accuracy_score(test_labels, np.argmax(probas_, axis=1))
    print("\n\tTest F1 weighted: ", f1)
    print("\tTest Accuraccy:", acc, "\n")
	
	
	# !!!
	# The glrp currently runs only for the number of the data points equal to or less than the batch size
    data_to_test = test_data[0:common["batch_size"], ]
    probas_ = model.get_probabilities(data_to_test)
    labels_by_network = np.argmax(probas_, axis=1)
    labels_data_to_test = test_labels[0:common["batch_size"], ]
    I = np.eye(10)
    labels_hot_encoded = I[labels_by_network]

    glrp = glrp_scipy.GraphLayerwiseRelevancePropagation(model, data_to_test, labels_hot_encoded)
    rel = glrp.get_relevances()[-1]  # getting the relevances corresponding to the input layer

    data_to_test = coarsening.perm_data_back(data_to_test, perm, FEATURE_NUM)
    rel = coarsening.perm_data_back(rel, perm, FEATURE_NUM)

    results_dir = './figures/'

    start_example = 9
    end_example = 17

    visualize_mnist.plot_numbers(data_to_test[start_example:end_example, ], rel[start_example:end_example, ],
                                 labels_data_to_test[start_example:end_example, ],
                                 labels_by_network[start_example:end_example, ], results_dir)

    # start_example = 9
    # end_example = 17
    for i in range(start_example, end_example):  # 9, 17
        heatmap = visualize_mnist.get_heatmap(rel[i,])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(heatmap, cmap='Reds', interpolation='bilinear')
        fig.savefig('{0}LRP_w^+_correct_label_index{1}_{2}.png'.format(results_dir, str(i), str(test_labels[i])))
