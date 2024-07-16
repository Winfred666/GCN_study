import pickle as pkl
import  scipy.sparse as sp
import sys

# networkx here is to convert the adjacent list to adjacency matrix.
import  networkx as nx
import numpy as np

from utils import sparse_to_tuple

def parse_index_file(filename):
    """
    Parse an index file and return a list of integers.
    The index file should contain one integer per line.

    Parameters:
    filename (str): The path to the index file.
    
    Returns:
    list[int]: A list of integers parsed from the file.
    """
    indices = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and convert the line to an integer
            index = int(line.strip())
            indices.append(index)
    return indices


def sample_mask(idx, l):
    """
    Create length-l mask on idx(1).
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)



# this is imported from outside.
def load_siteseer(dataset_str):
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
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/citeseer/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/citeseer/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # allx is the feature of all nodes add more node here 
    features = sp.vstack((allx, tx)).tolil()
    # prepare the adjacency matrix
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    adj = nx.to_scipy_sparse_matrix(nx.from_dict_of_lists(graph))

    # prepare the label, need to concat label of tx here.
    labels = np.vstack((ally, ty))
    # sort the label according to the test index.
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    NODE_NUM = labels.shape[0]
    idx_test = test_idx_range.tolist()
    test_mask = sample_mask(idx_test, NODE_NUM)

    # split the train and validate set five times here
    VALIDATE_NUM = int((len(y) + 500)/5)

    train_masks = []
    val_masks = []
    for i in range(5):
        idx_val = range(i*VALIDATE_NUM, (i+1)*VALIDATE_NUM)
        idx_train = list(set(range(VALIDATE_NUM*5)) - set(idx_val))
        train_masks.append(sample_mask(idx_train, NODE_NUM))
        val_masks.append(sample_mask(idx_val, NODE_NUM))
    
    # use five cross validation here
    return adj, features, labels, train_masks, val_masks, test_mask
