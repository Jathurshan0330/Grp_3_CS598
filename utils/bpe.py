import torch
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm   

def assign_to_nearest_cluster_batch(embeddings, cluster_centers):
    
    
    if not cluster_centers.is_cuda:
        # assign to embedding device
        cluster_centers = cluster_centers.to(embeddings.device)
    
    
    batch_size, time_steps, embedding_dim = embeddings.shape
    # num_clusters, _ = cluster_centers.shape

    # Reshape embeddings to (batch_size * time_steps, embedding_dim)
    # print(embeddings.shape)
    embeddings_flat = embeddings.view(-1, embedding_dim)  # Shape: (batch_size * time_steps, embedding_dim)

    # Compute squared Euclidean distances between embeddings and cluster centers
    # Using broadcasting to efficiently compute the distance
    distances = torch.cdist(embeddings_flat, cluster_centers, p=2)  # Shape: (batch_size * time_steps, num_clusters)

    # Get the index of the nearest cluster for each embedding
    nearest_clusters = torch.argmin(distances, dim=1)  # Shape: (batch_size * time_steps)
    
    # Reshape the result back to (batch_size, time_steps, 1)
    nearest_clusters = nearest_clusters.view(batch_size, time_steps)
    #convert nearest_clusters to list
    nearest_clusters =  nearest_clusters.cpu().numpy()#.astype(str)#.tolist()
    # nearest_clusters = nearest_clusters.cpu().numpy().tolist()
    
    # map each element to str
    # nearest_clusters = [[str(token_vocab[i]) for i in j] for j in nearest_clusters]

    return nearest_clusters



from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.token = None  # BPE key for matched sequence

def build_bpe_trie(bpe_vocab):
    root = TrieNode()
    for key, value in bpe_vocab.items():
        node = root
        value_list = value.strip().split(' ')
        for token in value_list:
            node = node.children[token]
        node.token = key  # Mark end of the BPE sequence
    return root

