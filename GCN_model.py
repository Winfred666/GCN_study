import  torch
from    torch import nn
from    torch.nn import functional as F

from GCN_layer import GraphConvolution_Base
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
        self.layers.add_module('layer0', GraphConvolution_Base(layer_dims[0], layer_dims[1], activation=F.relu, dropout_rate=dropout_rate,
            input_nonezero_num=input_nonezero_num, is_sparse_inputs=True))
        last_i = len(layer_dims)-2
        for i in range(1,last_i):
            # add all layers
            self.layers.add_module(f'layer{i}',
            GraphConvolution_Base(layer_dims[i], layer_dims[i+1], activation=F.relu, dropout_rate=dropout_rate))
        
        # the last layer will use softmax as activation function, suitable for classification
        self.layers.add_module(f'layer{last_i}',
            GraphConvolution_Base(layer_dims[last_i], layer_dims[last_i+1], activation=F.softmax, dropout_rate=dropout_rate))
        
    def forward(self, inputs):
        x,hat_A = inputs
        for layer in self.layers:
            # check all output after one layer
            # print(f'After layer: {x.sum()}')
            x = layer((x,hat_A))
        return x,hat_A

    # as the paper suggest , use L2 only at the update of first convolution layer.
    def L2_reg_layer1(self):
        layers = self.layers[:-1] # get all layers except the last one
        loss = None
        # add every sum of square of parameters
        for layer in layers:
            for p in layer.parameters():
                if loss is None:
                    loss = p.pow(2).sum()
                else:
                    loss += p.pow(2).sum()
        return loss
    
class GCN_FC(GCN_Base):
    def __init__(self, layer_dims, # essential for all model
    dropout_rate = 0.5, # normalization technique.
    input_nonezero_num = 0): # because the first input feature will probably be sparse
        super().__init__(layer_dims,dropout_rate,input_nonezero_num)
        #下面是新增全连接层，按照论文修改
        self.fc_layers = nn.Sequential()

    def forward(self, inputs):
        x,hat_A = inputs
        for layer in self.layers:
            # check all output after one layer
            # print(f'After layer: {x.sum()}')
            x = layer((x,hat_A))
        
        ###
        #需要修改x的形状才能作为全连接层的输入，之后再改回来
        #原先x是(num_nodes, in_feature_dim),分别代表每个图节点数,每个节点的特征向量维度
        #x = x.view(-1, self.layer_dims[-1])  # 将节点特征展开为 (batch_size * num_nodes, feature_dim)
        x = x.float()#原先的x和hat_A都是float64也就是double类型的,而全连接层的参数都是float32也就是float类型的,所以要转化一下
        x = self.fc_layers(x)
        x = x.double()#再变回float64
        #x = x.view(-1, hat_A.size(0), self.fc_dims[-1])  # 恢复形状为 (batch_size, num_nodes, output_dim)
        ###
        return x,hat_A

    # as the paper suggest , use L2 only at the update of first convolution layer.
    def L2_reg_layer1(self):
        layers = self.layers[:-1] # get all layers except the last
        # wander what to do with regulation loss in Full connectted layer.
        loss = None
        # add every sum of square of parameters
        for layer in layers:
            for p in layer.parameters():
                if loss is None:
                    loss = p.pow(2).sum()
                else:
                    loss += p.pow(2).sum()
        return loss