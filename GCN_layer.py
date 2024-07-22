
# define the convolutional layer of GCN
import  torch
from    torch import nn
from    torch.nn import functional as F
from utils import sparse_dropout

# 1.use classical conv layer deduced by normalize laplace matrix
class GraphConvolution_Base(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, activation, # essential for all layers.
                need_bias = False, 
                # 1. bias is optional, because dataset is simple, so reduce complexity
                # 2. normalized laplace matrix / divide N aggregation, also reduce bias.
                # 2. batch/layer normalization can do this work.
                dropout_rate = 0,  # normalization technique.
                is_sparse_inputs = False, input_nonezero_num = 0): # use flag to optimize sparse input like adjacency matrix
        
        super().__init__()
        
        # input dim is usually the number of features, F x N, so keep F.
        self.in_feature_dim = in_feature_dim
        # output dim: H x N, so keep H, and the weight matrix is H x F
        self.out_feature_dim = out_feature_dim
        self.activation = activation

        self.is_sparse_inputs = is_sparse_inputs
        self.input_nonezero_num  = input_nonezero_num
        self.weight = nn.Parameter(torch.randn(in_feature_dim, out_feature_dim,dtype=float))
        if need_bias:
            self.bias = nn.Parameter(torch.zeros(out_feature_dim,dtype=float))
        else:
            self.bias = None
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate) # dropout layer

        self.init_parameters()


    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight) # init weight matrix with xavier uniform
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, inputs):
        # here hat_A is calculate from tilde_D^(-1/2) @ tilde_A @ tilde_D^(-1/2)
        x, hat_A = inputs 

        # kill some forwarding feature from last layer to prevent overfitting
        # x as input graph, when embedding vector has too much zero, it's also sparse.
        if self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout_rate, self.input_nonezero_num)
        else:
            x = self.dropout(x)

        # x: F x N, follow the formula to calculate H_{l+1}
        if self.is_sparse_inputs:
            xw = torch.spmm(x, self.weight)
        else:
            xw = torch.mm(x, self.weight)
        
        # print(hat_A.shape,x.shape,self.weight.shape)
        # in the paper this is an efficient function to calculate sparse matrix multiplication.
        # and adj matrix is always sparse
        output = torch.spmm(hat_A, xw)

        if self.bias is not None:
            output = output + self.bias

        # finally apply activation function
        if(self.activation == F.softmax):
            output = self.activation(output, dim=1)
        else:
            output = self.activation(output)
        return output

# this is a fully connected layer that run on sparse cuda, and use dropout
class SparseLinear(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, activation
        , dropout_rate = 0, is_sparse_inputs = False, input_nonezero_num = 0):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.weight = nn.Parameter(torch.randn(in_feature_dim, out_feature_dim,dtype=float))
        self.bias = nn.Parameter(torch.zeros(out_feature_dim,dtype=float))
        self.is_sparse_inputs = is_sparse_inputs
        self.input_nonezero_num = input_nonezero_num
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, inputs):
        x = inputs
        if self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout_rate, self.input_nonezero_num)
            xw = torch.spmm(x, self.weight)
        else:
            x = self.dropout(x)
            xw = torch.mm(x, self.weight)
        output = xw + self.bias
        if self.activation == F.softmax:
            output = self.activation(output, dim=1)
        else:
            output = self.activation(output)
        return output