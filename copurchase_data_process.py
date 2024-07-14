# 特征工程，利用无向图的邻接矩阵信息，搭建每一个矩阵的内嵌特征向量。
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
from node2vec import Node2Vec
import networkx as nx

# 1. 读取图数据
def read_graph(file_path):
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
            if i < max_id and j < max_id:  # 仅处理小于max_id的边
                graph[i].append(j)
                graph[j].append(i)
                
            #限制一下读取数目
            #cnt+=1
            #if cnt >= 100000:
            #    break
    return graph

# 2. 读取社区数据
def read_community(file_path):
    communities = []
    cnt=0
    with open(file_path, 'r') as f:
        for line in f:
            community = list(map(int, line.strip().split()))
            # 仅保留小于max_id的节点
            filtered_community = [node for node in community if node < max_id]
            if filtered_community:  # 确保非空
                communities.append(filtered_community)
                
            #限制一下读取数目
            #cnt+=1
            #if cnt >= 20000:
            #    break
    return communities
    
# 3. 构建特征向量（使用 Node2Vec）
def build_feature_vectors(graph):
    # 将图转换为真正的图
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # 创建 Node2Vec 模型
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=20, workers=8)
        #edges: 上一步生成的边列表。dimensions: 特征向量的维度。walk_length: 每次随机游走的长度。num_walks: 每个节点进行随机游走的次数。workers: 用于并行计算的线程数。
    
    # 训练模型
    model = node2vec.fit(window=10, min_count=1, batch_words=100)
        #window: 上下文窗口的大小，影响训练时考虑的邻居范围。min_count: 忽略出现次数少于该值的节点，确保每个节点至少出现一次。batch_words: 一次处理的节点数量。
    
    # 获取节点特征向量
    #features = np.array([model.wv[str(node)] for node in range(1,max_id)])  
        #model.wv 是训练好的词向量（特征向量）的集合。通过列表推导式，获取从 0 到 max_id 的每个节点的特征向量，并将其存储为 NumPy 数组。
        
    num_nodes = max_id
    dimensions = 64
    features = np.zeros((num_nodes, dimensions))

    for node in G.nodes():
        features[node] = model.wv[str(node)]
    return sp.csr_matrix(features)  # 转换为稀疏矩阵
