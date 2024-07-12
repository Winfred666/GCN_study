import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from GCN_model import GCN_Base
from utils import calculate_hat_A, masked_loss, masked_acc, preprocess_features

# here should set some parameter for training
# @data: the data for training, including adj , x(features), y(label), mask(indicate which is training data)
# adj: N*N sparse matrix, x: N*F sparse matrix, label: N vector, train_mask: N vector, loss_mask: N vector
def train_base(layer_dims, data,
            dropout_rate = 0.2,learning_rate = 0.01, weight_decay = 0.0001, # some hyperparameters
            epoch_num = 200,
            self_importance = 1.0, laplace_norm = True,
            device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'):

    
    print(f'Using device: {device} args: layer_dims: {layer_dims}, dropout_rate: {dropout_rate}, \n\
        learning_rate: {learning_rate}, weight_decay: {weight_decay}, epoch_num: {epoch_num}, \n\
        self_importance: {self_importance}, laplace_norm: {laplace_norm}')
    
    data = data_to_tensor(data, device)
    
    # now get hat_A and num_features_nonzero from data.
    adj, x, y, train_mask, validate_mask = data

    # get hat_A, and adj is no more needed.
    # WARNING: A is only a local adjacent matrix, because only input X as it.
    hat_A = calculate_hat_A(adj,self_importance,laplace_norm) 
    # send hat_A to device: see sparse_to_tuple for more info
    i = torch.from_numpy(hat_A[0]).long().to(device) # 0 is coordinate of non-zero element
    v = torch.from_numpy(hat_A[1]).to(device) # 1 is none zero value
    hat_A = torch.sparse_coo_tensor(i.t(), v, hat_A[2], dtype=float, device=device) # 2 is shape of matrix

    
    input_nonezero_num = x._nnz() # get the number of none zero element in input feature(sparse)
    net = GCN_Base(layer_dims, dropout_rate, input_nonezero_num)

    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    net.train()
    
    lost_list = []
    val_acc_list = []

    for epoch in range(epoch_num):
        # every time, input feature of all node in graph, no minibatch
        # in fact we can use minibatch, only sample some node to train.
        out = net((x, hat_A))[0] # get the y predict output, instead of hat_A
        
        loss = masked_loss(out, y, train_mask)
        loss += weight_decay * net.L2_reg_layer1() # add L2 regularization
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        # Check gradients
        # for param in net.parameters():
        #     if param.grad is not None:
        #         print(param.grad.data.sum())
        
        out = net((x, hat_A))[0]
        acc = masked_acc(out, y, train_mask) # get accuracy on train set
        acc_val = masked_acc(out, y, validate_mask) # get accuracy on validate set
        val_acc_list.append(acc_val)
        lost_list.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Train Set Acc: {acc.item():.4f}, Validate Set Acc: {acc_val.item():.4f}')
        
    return net, val_acc_list, lost_list




def data_to_tensor(data, device):
    adj, x, y, train_mask, validate_mask = data

    # help me to output the value in the sparse matrix x
    # print(x)
    # train_x = feature_x[train_mask]
    # validate_x = feature_x[validate_mask]
    
    validate_mask = torch.from_numpy(validate_mask.astype(int)).to(device)
    train_mask = torch.from_numpy(train_mask.astype(int)).to(device)
    

    def to_tenser_sparse(feature_x):
        feature_x = preprocess_features(feature_x)
        i = torch.from_numpy(feature_x[0]).long().to(device)
        v = torch.from_numpy(feature_x[1]).to(device)
        return torch.sparse_coo_tensor(i.t(), v, feature_x[2], dtype=float, device=device)
    
    x = to_tenser_sparse(x)
    
    # print(x.sum())

    y = torch.from_numpy(y).long().to(device)

    return adj, x, y, train_mask, validate_mask
