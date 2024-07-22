# use seaborn to see 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# 绘制验证准确率曲线
def plot_acc_loss(test_acc_list, train_acc_list, loss_list,two_graph=True):
    if two_graph:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
    else:
        plt.figure(figsize=(6, 5))
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend() #加图例

    # 绘制损失值曲线
    if two_graph:
        plt.subplot(1, 2, 2)
        plt.plot(loss_list, label='Training Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

    plt.tight_layout() #调整子图布局
    plt.show()


# convert feature to dense matrix
def embedding_to_2d(features,feature_num = 50):
    node_embeddings = np.asarray(features)
    # Use t-SNE for dimensionality reduction, only x and y feature.
    tsne = TSNE(n_components=2, random_state=0)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)
    pca = PCA(n_components=feature_num)
    # Optionally, use PCA before t-SNE for better results
    node_embeddings_pca = pca.fit_transform(node_embeddings)
    node_embeddings_2d = tsne.fit_transform(node_embeddings_pca)
    return node_embeddings_2d

def plot_cluster(node_embeddings_2d,labels):
    # Get cluster labels (assuming you have them from a clustering algorithm)
    cluster_labels = labels
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("hsv", len(set(cluster_labels)))  # Choose a color palette
    sns.scatterplot(x=node_embeddings_2d[:, 0], y=node_embeddings_2d[:, 1], hue=cluster_labels, palette=palette, 
        legend=False,)
    # plt.title('t-SNE Visualization of Classification in Citeseer()')
    plt.show()

def plot_graph(adj,labels):
    G = nx.from_scipy_sparse_matrix(adj)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)
    # filter out the don't care class
    hide_nodes = np.where(labels == -1)[0]
    # delete the nodes
    G.remove_nodes_from(hide_nodes)
    labels = np.delete(labels, hide_nodes, axis=0)
    nx.draw(G, pos=pos, node_color=labels, cmap='tab20', node_size=8, with_labels=False)
    # Draw the edges
    plt.show()