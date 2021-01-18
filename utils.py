import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.io as sci
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data1(dataset_str):
    """
    Loads input data from gcn/data directory

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    dir = 'C:\\Users\\rmmpq\\Desktop\\EEG-DL-master\\Preprocess_EEG_Data\\For-GCN-based-Models\\10\\'
    # Read the Adjacency matrix
    Adjacency_Matrix = pd.read_csv(dir + 'Adjacency_Matrix.csv', header=None)
    Adjacency_Matrix = np.array(Adjacency_Matrix).astype('float32')
    Adjacency_Matrix = sp.csr_matrix(Adjacency_Matrix)

    import h5py
    f = h5py.File(dir + 'training_set_1.mat', 'r')
    test = f.get('training_set1')
    test = np.array(test)  # For converting to a NumPy array
    test = test.transpose((2, 1, 0))
    test_set = list()
    for item in test:
        temp = sp.lil_matrix(item)
        test_set.append(temp)

    f_train = h5py.File(dir + 'test_set_1.mat', 'r')
    train = f_train.get('test_set1')
    train = np.array(train)
    train = train.transpose((2, 1, 0))
    train_set = list()
    for item_train in train:
        temp_train = sp.lil_matrix(item_train)
        train_set.append(temp_train)

    f_test_label = h5py.File(dir + 'training_label_1.mat', 'r')
    test_labell = f_test_label.get('label_training1')
    test_labell = np.array(test_labell)
    test_labell = test_labell.transpose((1, 0))
    test_label = list()
    for item_test_label in test_labell:
        temp_test_label = sp.lil_matrix(item_test_label)
        test_label.append(temp_test_label)

    f_train_label = h5py.File(dir + 'test_label_1.mat', 'r')
    train_labell = f_train_label.get('label_test1')
    train_labell = np.array(train_labell)
    train_labell = train_labell.transpose((1, 0))
    train_label = list()
    for item_train_label in train_labell:
        temp_train_label = sp.lil_matrix(item_train_label)
        train_label.append(temp_train_label)

    idx_train = range(train.shape[0])
    idx_test = range(test.shape[0])
    train_mask = sample_mask(idx_train, len(train_label))
    test_mask = sample_mask(idx_test, len(test_label))
    return Adjacency_Matrix, train_set, test_set, train_labell, test_labell, train_mask, test_mask


def load_data2(dataset_str):
    """
    Loads input data from gcn/data directory

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    import h5py
    dir = 'C:\\Users\\rmmpq\\Desktop\\gcn-master\\gcn\\For-GCN-based-Models\\four_PLI_4_20\\'
    # Read the Adjacency matrix train
    f_adj_train = h5py.File(dir + 'pli_graph_train.mat', 'r')
    adj_train = f_adj_train.get('pli_graph_train')
    adj_train = np.array(adj_train).astype('float32')  # For converting to a NumPy array
    adj_train = adj_train.transpose((2, 0, 1))

    # Read the Adjacency matrix test
    f_adj_test = h5py.File(dir + 'pli_graph_test.mat', 'r')
    adj_test = f_adj_test.get('pli_graph_test')
    adj_test = np.array(adj_test).astype('float32')  # For converting to a NumPy array
    adj_test = adj_test.transpose((2, 0, 1))

    f = h5py.File(dir + 'training_set_1.mat', 'r')
    train = f.get('training_set1')
    train = np.array(train).astype('float32')  # For converting to a NumPy array
    train = train.transpose((2, 1, 0))

    f_test = h5py.File(dir + 'test_set_1.mat', 'r')
    test = f_test.get('test_set1')
    test = np.array(test).astype('float32')
    test = test.transpose((2, 1, 0))

    #------------------------------------------------------------------------
    dir1 = 'C:\\Users\\rmmpq\\Desktop\\gcn-master\\gcn\\For-GCN-based-Models\\four_PLI_4_20\\'

    f_train_label = h5py.File(dir + 'training_label_1.mat', 'r')
    train_labell = f_train_label.get('label_training1')
    train_labell = np.array(train_labell).astype('float32')
    train_labell = train_labell.transpose((1, 0))

    f_test_label = h5py.File(dir + 'test_label_1.mat', 'r')
    test_labell = f_test_label.get('label_test1')
    test_labell = np.array(test_labell).astype('float32')
    test_labell = test_labell.transpose((1, 0))

    adj_train_set = list()
    for adj_item_train in adj_train:
        temp = sp.lil_matrix(adj_item_train)
        adj_train_set.append(temp)

    adj_test_set = list()
    for adj_item_test in adj_test:
        temp = sp.lil_matrix(adj_item_test)
        adj_test_set.append(temp)

    test_set = list()
    for item in test:
        temp = sp.lil_matrix(item)
        test_set.append(temp)
    # val_set = test_set[-100: -1]

    train_set = list()
    for item_train in train:
        temp_train = sp.lil_matrix(item_train)
        train_set.append(temp_train)

    test_label = list()
    for item_test_label in test_labell:
        temp_test_label = sp.lil_matrix(item_test_label)
        test_label.append(temp_test_label)
    # val_label = test_labell[-100: -1]

    train_label = list()
    for item_train_label in train_labell:
        temp_train_label = sp.lil_matrix(item_train_label)
        train_label.append(temp_train_label)

    idx_train = range(train.shape[0])
    # idx_val = range(len(val_set))
    idx_test = range(test.shape[0])

    train_mask = sample_mask(idx_train, len(idx_train))
    test_mask = sample_mask(idx_test, len(idx_test))

    return adj_train, adj_test, train_set, train_labell, test_set, test_labell, train_mask, test_mask

def rettifyLabel(excludeList, train_labell, test_labell, train, test, adj_train, adj_test):
    argMaxTrain = train_labell.argmax(1)
    argMaxTest = test_labell.argmax(1)
    tr2 = np.where(argMaxTrain == 2)
    te2 = np.where(argMaxTest == 2)
    i = np.delete(train_labell, tr2, 0)

    return 0


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    fea = list()
    for prove in features:
        rowsum = np.array(prove.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        prove = r_mat_inv.dot(prove)
        fea.append(sparse_to_tuple(prove))

    return fea


def preprocess_features1(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj2(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    fea = list()
    for prove in adj:
        adj_normalized = normalize_adj(prove + sp.eye(prove.shape[0]))
        fea.append(sparse_to_tuple(adj_normalized))
    return fea


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

#PLI

# 42,26 -> 0 - 42,85 -> 1 - 41,28 -> 2 - 45,83 -> 3 -       -> 4 =><= 2 strati - 4
# 42,85 -> 0 - 44,64 -> 1 - 45,23 -> 2 - 41,07 -> 3 -       -> 4 =><= 1 strati - 4

# 43,45 -> 0 - 45,83 -> 1 - 48,21 -> 2 - 47.02 -> 3 - 36.90 -> 4 =><= 1 strati - 1
# 42,26 -> 0 - 41,07 -> 1 - 44,08 -> 2 - 45,23 -> 3 - 36,86 -> 4 =><= 2 strati - 1

# 46.42 -> 0 - 41.66 -> 1 - 39.28 -> 2 - 39.28 -> 3 - 32.73 -> 4 =><= 1 strati 2 convoluzioni - 1

# Pearson
# 32.73 -> 0 - 35.71 -> 1 - 36,90 -> 2 - 32,73 -> 3 -  -> 4 =><= 1 strati - 1
# 33.33 -> 0 - 36.90 -> 1 - 38.69 -> 2 - 29.76 -> 3 -  -> 4 =><= 1 strati - 4

# 33.33 -> 0 - 40.47 -> 1 - 41.66 -> 2 - 34.52 -> 3 -  -> 4 =><= 2 strati - 4
# 38.09 -> 0 - 41.66 -> 1 - 36.90 -> 2 - 35.71 -> 3 -  -> 4 =><= 2 strati - 1

def construct_feed_dict2(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def construct_feed_dict3(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    support = [support]
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
