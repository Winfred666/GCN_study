import torch
from torch.nn import functional as F
import  scipy.sparse as sp
import numpy as np

# a utility function to drop out some features from input Sparse vector in GPU
def sparse_dropout(x, rate, noise_shape):
    """
    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask] # transpose matrix
    v = v[dropout_mask] # only get remained node

    out =  torch.sparse_coo_tensor(i, v, x.shape, dtype=float, device=x.device)

    out = out * (1./ (1-rate))

    return out


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
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


# the hat A is essential for GCN, and unchanged during the training process
# so only calculate it once before training
# remember that adj is sparse matrix
def calculate_hat_A(adj,self_importance=1.0,laplace_norm=True):
    tilde_A = adj + self_importance * (np.eye(adj.shape[0]))
    # get degree matrix
    tilde_D = np.array(adj.sum(1))
    hat_A = None
    if laplace_norm:
        D_m2 = np.power(tilde_D, -0.5).flatten()
        D_m2[np.isinf(D_m2)] = 0.0
        D_m2 = np.diag(D_m2)
        # get normalized laplace matrix
        hat_A = D_m2.T @ tilde_A @ D_m2 # D^-0.5AD^0.5
    else:
        D_inv = torch.power(tilde_D, -1)
        D_inv[np.isinf(D_inv)] = 0.0
        D_inv = torch.diag(D_inv)
        hat_A = D_inv.T @ tilde_A
    
    return sparse_to_tuple(sp.coo_matrix(hat_A)) # first transform back to sparse matrix


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    just make the row sum to 1.
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]

    r_inv = np.divide(1, rowsum, out=np.zeros_like(rowsum), 
    where=rowsum!=0).flatten()
    # r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    # r_inv[np.isinf(r_inv)] = 0. # zero inf data

    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features) # [coordinates, data, shape], []



# using cross entrophy loss
def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean() # in fact this make all element in mask to 1.
    loss *= mask
    loss = loss.mean()
    return loss

# calculate accuracy
def masked_acc(out, label, mask):
    pred = out.argmax(dim=1) # get prediction from output, using max probability.
    # print(out)
    correct = torch.eq(pred, label).float() # get correct prediction
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


# in confusion matrix, the first dimension(row) is predict label, the second(each column) is actual label
def micro_F1(confusion_matrix):
    # first need to calculate FP,TP,FN,TN using each class as positive class
    TP_sum, FP_sum, FN_sum, TN_sum = 0,0,0,0
    for i in range(confusion_matrix.shape[0]): # for every class
        TP_sum += confusion_matrix[i,i] # always on the diagonal
        FN_sum += confusion_matrix[:,i].sum() - confusion_matrix[i,i] # the sum of column i(real label is P) - TP
        FP_sum += confusion_matrix[i,:].sum() - confusion_matrix[i,i] # the sum of row i(predict label is P) - TP
        TN_sum += confusion_matrix.sum() - TP_sum - FP_sum - FN_sum 
    # then calculate precision, recall and F1
    precision = TP_sum / (TP_sum + FP_sum)
    recall = TP_sum / (TP_sum + FN_sum)
    F1 = 2 * precision * recall / (precision + recall)
    return F1