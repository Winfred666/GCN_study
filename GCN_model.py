import  torch
from    torch import nn
from    torch.nn import functional as F

from GCN_layer import GraphConvolution_Base, SparseLinear
# the whole model of n layers convolution layer GCN,
# test layer n to find over-smoothing problem

class GCN_Base(nn.Module):
    def __init__(self, layer_dims, # essential for all model
    dropout_rate = 0.5, # normalization technique.
    input_nonezero_num = 0): # because the first input feature will probably be sparse
        super().__init__()
        self.layer_dims = layer_dims
        # the second convolution layer may not be sparse, so here use this kind of settings.
        # now need to add many layers, so use a list to store all layers.
        self.layers = nn.Sequential()
        
        # remember only the first layer need to be sparse, so the first layer is different from others.
        last_i = len(layer_dims)-2
        if(last_i > 0):
            self.layers.add_module('layer0', GraphConvolution_Base(layer_dims[0], layer_dims[1], activation=F.relu, dropout_rate=dropout_rate,
                input_nonezero_num=input_nonezero_num, is_sparse_inputs=True))
            for i in range(1,last_i):
                # add all layers
                self.layers.add_module(f'layer{i}',
                GraphConvolution_Base(layer_dims[i], layer_dims[i+1], activation=F.relu, dropout_rate=dropout_rate))
            
            # the last layer will use softmax as activation function, suitable for classification
            self.layers.add_module(f'layer{last_i}',
                GraphConvolution_Base(layer_dims[last_i], layer_dims[last_i+1], activation=F.softmax, dropout_rate=dropout_rate))
        else:
            self.layers.add_module('layer0', GraphConvolution_Base(layer_dims[0], layer_dims[1], activation=F.softmax, dropout_rate=dropout_rate,
                input_nonezero_num=input_nonezero_num, is_sparse_inputs=True))
        
    def forward(self, inputs):
        x,hat_A = inputs
        for layer in self.layers:
            x = layer((x,hat_A))
        return x,hat_A

    # as the paper suggest , use L2 only at the update of first convolution layer.
    def L2_reg_layer1(self):
        if(len(self.layers) == 1):
            layers = self.layers
        else:
            layers = self.layers[:-1] # get all layers except the last one
        loss = 0
        # add every sum of square of parameters
        for layer in layers:
            for p in layer.parameters():
                loss += p.pow(2).sum()
        return loss
    

# GCN that add two full connected layer after the last convolution layer
class GCN_FC(GCN_Base):
    def __init__(self, layer_dims, # essential for all model
    fc_dims, #定义全连接层的尺寸
    dropout_rate = 0.5, # normalization technique.
    input_nonezero_num = 0): # because the first input feature will probably be sparse
        super().__init__(layer_dims,dropout_rate,input_nonezero_num)
        self.layers[-1].activation = F.relu # change the activation function of last layer to relu
        #下面是新增全连接层，按照论文修改
        self.fc_dims = fc_dims
        self.fc_layers = nn.Sequential()
        #第一层
        # 全连接层也要加入 dropout，但由于是 sparse cuda, 只能手动添加
        self.fc_layers.add_module('fc_layer0', SparseLinear(layer_dims[-1], fc_dims[0], activation=F.relu, 
            dropout_rate=dropout_rate))
        #后几层
        last_i = len(fc_dims)-2
        for i in range(0, last_i):
            self.fc_layers.add_module(f'fc_layer{i}', SparseLinear(fc_dims[i], fc_dims[i+1], activation=F.relu, dropout_rate=dropout_rate))
        
        self.fc_layers.add_module(f'fc_layer{len(fc_dims)}', 
            SparseLinear(fc_dims[last_i],fc_dims[last_i+1], activation=F.softmax)) #最后一层用softmax，且没有 drop out
        
    def forward(self, inputs):
        x , hat_A = super().forward(inputs)
        x = self.fc_layers(x)
        return x,hat_A

    # as the paper suggest , use L2 only at the update of first convolution layer.
    def L2_reg_layer1(self):
        # wander what to do with regulation loss in Full connectted layer.
        loss = 0
        # add every sum of square of parameters
        for layer in self.layers:
            for p in layer.parameters():
                loss += p.pow(2).sum()
        for layer in self.fc_layers[:-1]: # get all layers except the last
            for p in layer.parameters():
                loss += p.pow(2).sum()
        return loss

class GCN_Simplify(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, feature_dim, class_dim):
        super().__init__()
        self.W = nn.Linear(feature_dim, class_dim,dtype=float)

    def forward(self, x):
        x,hat_A = x
        return self.W(x),hat_A
    
    def L2_reg_layer1(self):
        loss = 0
        for p in self.W.parameters():
            loss += p.pow(2).sum()
        return loss