#!python

"""
The script performs training of the Keras version of the GCNN model
on the same data as it is done in the script "run_glrp_ge_data_record_relevances.py".
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split

from components import nn_cnn_models, data_handling#, glrp_keras, nn_cnn_evaluation
from lib import graph, coarsening
from sklearn.utils import class_weight
# from sklearn.model_selection import StratifiedKFold

import time

rndm_state = 7
np.random.seed(rndm_state)
import tensorflow as tf


if __name__ == "__main__":

    path_to_feature_val = "./data/GE_PPI/GEO_HG_PPI.csv"
    path_to_feature_graph = "./data/GE_PPI/HPRD_PPI.csv"
    path_to_labels = "./data/GE_PPI/labels_GEO_HG.csv"

    # path_to_feature_val = "./data/BRCA_subtypes/HPRD_edges_normalized_data_filtered.csv"
    # path_to_feature_graph = "./data/BRCA_subtypes/HPRD_edges_filtered_network.csv"
    # path_to_labels = "./data/BRCA_subtypes/output_label_file.csv"

    DP = data_handling.DataPreprocessor(path_to_feature_values=path_to_feature_val, path_to_feature_graph=path_to_feature_graph,
                                        path_to_labels=path_to_labels)
    X = DP.get_feature_values_as_np_array()  # gene expression
    A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # adjacency matrix of the PPI network
    y = DP.get_labels_as_np_array()  # labels

    print("GE data, X shape: ", X.shape)
    print("Labels, y shape: ", y.shape)
    print("PPI network adjacency matrix, A shape: ", A.shape)

    X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Need to know which patients got into train and test subsets
    _, _, patient_indexes_train, patient_indexes_test = train_test_split(X, DP.labels.columns.values.tolist(), test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Data frame with test patients and corresponding ground truth labels
    patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})

    # !!!
    # Making data lying in the interval [0, 8.35]
    X_train = X_train_unnorm - np.min(X)
    X_test = X_test_unnorm - np.min(X)

    print("X_train max", np.max(X_train))
    print("X_train min", np.min(X_train))
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train, shape: ", y_train.shape)
    print("y_test, shape: ", y_test.shape)

    # coarsening the PPI graph to perform pooling in the model
    graphs, perm = coarsening.coarsen(A, levels=2, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    X_train = coarsening.perm_data(X_train, perm)
    X_test = coarsening.perm_data(X_test, perm)

    # !!!
    # No coarsening here
    # L = [graph.laplacian(A, normalized=True) for i in range(2)]

    n_train = X_train.shape[0]

    params = dict()
    params['dir_name'] = 'GE'
    params['num_epochs'] = 60
    params['batch_size'] = 109
    params['eval_frequency'] = 40

    # Building blocks.
    params['filter'] = 'chebyshev5'
    params['brelu'] = 'b1relu'
    params['pool'] = 'mpool1'

    # Number of classes.
    C = y.max() + 1
    assert C == np.unique(y).size

    # Architecture.
    params['F'] = [32, 32]  # Number of graph convolutional filters.
    params['K'] = [8, 8]  # Polynomial orders.
    params['p'] = [2, 2]  # Pooling sizes.
    params['M'] = [512, 128, C]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 1e-4
    params['dropout'] = 1
    params['learning_rate'] = 1e-3
    params['decay_rate'] = 0.95
    params['momentum'] = 0
    params['decay_steps'] = n_train / params['batch_size']

    # !!!
    # Additional parameters needed
    params["L"] = L

    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(y_train),
    #                                                   y_train)
    # class_weights = dict(enumerate(class_weights))
    # print(class_weights)

    # MyChenNet is a wrapper for the ChebNet model
    # TODO: improve the functioning of the MyChebNet class
    # !!!
    # The model with biases constraints
    #model = nn_cnn_models.get_bias_constrained_cheb_net_model(**params)

    # to build the models as in the Genome Medicine paper
    model = nn_cnn_models.get_cheb_net_model(**params)
    my_cheb_net_for_cv = nn_cnn_models.MyChebNet(params, model)
    my_cheb_net_for_cv.create(feature_number=None)

    # model.summary()

    start = time.time()
    my_cheb_net_for_cv.fit(x=np.expand_dims(X_train, 2), y=y_train, validation_data=[np.expand_dims(X_test, 2), y_test],
                           verbose=1)
    end = time.time()
    print("\n\tTraining time:", end-start, "\n")
    # y_preds = my_cheb_net_for_cv.predict(X_test)

    y_preds = my_cheb_net_for_cv.predict(np.expand_dims(X_test, 2))
    acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
    f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
    print("test Accuraccy: %0.4f, test F1: %0.4f" % (acc, f1))


    # TODO: Keras version of GLRP
    # C = np.unique(y).shape[0]
    # I = np.eye(C)
    # y_train = I[y_train]
    # y_test = I[y_test]
    #
    # tf.keras.backend.clear_session()
    # glrp = glrp_keras.GraphLayerwiseRelevancePropagation(my_cheb_net_for_cv.model, X_test, y_test, L=params["L"][0:2], K=params['K'], p=params['p'])
    #
    # # # rel = glrp.get_relevances()[-1][:X_test.shape[0], :]
    # rel = glrp.get_relevances()
    # print(type(rel))
    # print(rel.shape)
    # print(rel.sum(axis=1))
    # rel = coarsening.perm_data_back(rel, perm, X.shape[1])
    # print(rel.shape)
    # print(rel.sum(axis=1))
