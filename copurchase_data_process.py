# 特征工程，利用无向图的邻接矩阵信息，搭建每一个矩阵的内嵌特征向量。
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from node2vec import Node2Vec
import networkx as nx

# 1. 读取图数据
def read_adjacent_mat(file_path):
    graph = defaultdict(list)
        #创建一个 defaultdict，其中的每个键的默认值是一个空列表（list）。
        #当你尝试访问一个未定义的键时，它会自动创建这个键，并将其值设置为一个空列表。
    cnt=0
    with open(file_path, 'r') as f:
        for line in f:
            i, j = map(int, line.strip().split()) 
                #strip()：去掉行首和行尾的空白字符（包括空格和换行符）
                #split()：将去掉空白后的字符串按空格分割成一个字符串列表。
                #map(int, ...)：将字符串列表中的每个字符串转换为整数。
            graph[i].append(j)
            graph[j].append(i)
    # change to sparse matrix
    adj = nx.to_scipy_sparse_matrix(nx.from_dict_of_lists(graph))
    return adj


# 2. 读取商品类别数据，作为标签使用（Ground Truth Label）
# class_num 表示需要学习的标签类别数，
# class_max_rate 表示 GCN 每个类最多的训练样本占比，考察正常训练情况
# class_max_samp 表示是否要求每个类别都有训练样本，用于考察类别不平衡的情况
# missing_class 表示一些没有训练集的样本，考察完全缺失类别的情况
def read_label(file_path,node_num,class_num = 500, class_max_rate = 0.2, class_max_samp = 0, missing_class = 0):
    # 由于商品类别过多，不利于 GCN 性能，因此只选择最多商品的 “大类” 作为分类依据
    cmtys = []
    labels = np.ones(node_num, dtype=int)*1 # 示未知类别，真值缺失，几个 mask 不应该包含 这些 项
    with open(file_path, 'r') as f:
        # index 即表示该商品的类别
        for line in f:
            community = list(map(int, line.strip().split()))
            # 仅保留小于node_num的节点
            filtered_community = [node for node in community if node <= node_num]
            cmtys.append(filtered_community)
    # 选择最多商品的 “大类” 作为分类依据
    cmtys.sort(key=lambda x: len(x), reverse=True)
    cmtys = cmtys[:class_num]
    print('Items belongs to the first class: ',len(cmtys[0]))
    print('Items belongs to the last class: ',len(cmtys[-1]))
    # 选择每个类别的训练样本
    train_mask = np.zeros(len(labels), dtype=int)
    test_mask = np.zeros(len(labels), dtype=int)
    for i, c in enumerate(cmtys):
        samp_num = 0
        if class_max_samp > 0:
            samp_num = int(min(len(c)*class_max_rate,class_max_samp))
        else:
            samp_num = int(len(c)*class_max_rate)
        samp_num = max(samp_num,1) # 每个类别至少有一个训练样本
        if(len(cmtys) - i <= missing_class):
            samp_num = 0 # 人为制造缺失类别
        for index,node in enumerate(c):
            ni = node-1
            if index < samp_num: # 划分训练样本
                train_mask[ni] = 1
            else:
                test_mask[ni] = 1
            labels[ni] = i+1 # 从 1 开始的类别标签
    return labels, train_mask, test_mask

# 3. 构建特征向量（使用 Node2Vec）
def build_feature_vectors(sp_adj,num_nodes,dimensions=500):
    # 将图转换为真正的图
    G = nx.from_scipy_sparse_matrix(sp_adj)
    
    print("Trainslate to networkx graph")
    
    # 创建 Node2Vec 模型
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=10, num_walks=5, workers=20)
        #edges: 上一步生成的边列表。 dimensions: 特征向量的维度。walk_length: 每次随机游走的长度。num_walks: 每个节点进行随机游走的次数。workers: 用于并行计算的线程数。
    print("Build Node2Vec model")
    # 训练模型
    model = node2vec.fit(window=6, min_count=1, batch_words=64)
        #window: 上下文窗口的大小，影响训练时考虑的邻居范围。min_count: 忽略出现次数少于该值的节点，确保每个节点至少出现一次。batch_words: 一次处理的节点数量。
    print("Finish feature random walk")

    # 获取节点特征向量
    #features = np.array([model.wv[str(node)] for node in range(1,max_id)])  
        #model.wv 是训练好的词向量（特征向量）的集合。通过列表推导式，获取从 0 到 max_id 的每个节点的特征向量，并将其存储为 NumPy 数组。
    
    features = np.zeros((num_nodes, dimensions))

    for node in G.nodes():
        features[node] = model.wv[str(node)]

    print("Extract the features out")
    return sp.csr_matrix(features)  # 转换为稀疏矩阵


def preprocess_graph(adj,labels,train_mask,test_mask, keep_percent=0.1):
    # 记录原始的节点编号
    node_id_list = np.arange(len(labels)) + 1
    # 对于不属于 mask 的闲杂节点，只保留 keep_percent 百分比
    
    remove_nodes = np.where(train_mask+test_mask == 0)[0]
    # 随机挑选剩下的节点
    remove_nodes = np.random.choice(remove_nodes, int(len(remove_nodes)*(1-keep_percent)), replace=False)
    print("Remove nodes: ",len(remove_nodes))
    rest_node = np.ones(len(labels), dtype=bool)
    rest_node[remove_nodes] = False

    adj = adj[rest_node][:,rest_node]
    labels = labels[rest_node]
    train_mask = train_mask[rest_node]
    test_mask = test_mask[rest_node]
    node_id_list = node_id_list[rest_node]

    # 删除孤立点，因为无法通过邻接矩阵学习到任何信息
    adj = nx.from_scipy_sparse_matrix(adj)
    iso_list = list(nx.isolates(adj))
    print("Remove isolated nodes: ",len(iso_list))
    adj.remove_nodes_from(iso_list)
    labels = np.delete(labels, iso_list, axis=0)
    train_mask = np.delete(train_mask, iso_list, axis=0)
    test_mask = np.delete(test_mask, iso_list, axis=0)
    node_id_list = np.delete(node_id_list, iso_list, axis=0)
    
    adj = nx.to_scipy_sparse_matrix(adj)
    return adj,labels,train_mask,test_mask,node_id_list