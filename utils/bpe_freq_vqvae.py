# import torch
# import numpy as np
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm   

# def assign_to_nearest_cluster_batch(embeddings, cluster_centers):
#     """
#     Assign each embedding in a batch to the nearest cluster based on Euclidean distance using GPU.

#     Args:
#     embeddings (torch.Tensor): A 3D tensor of shape (batch_size, time_steps, embedding_dim) representing the batch of embeddings.
#     cluster_centers (torch.Tensor): A 2D tensor of shape (n_clusters, embedding_dim) representing the cluster centers.

#     Returns:
#     torch.Tensor: A 3D tensor of shape (batch_size, time_steps, 1) where each value is the index of the nearest cluster.
#     """
#     # Move data to GPU if not already there
#     # if not embeddings.is_cuda:
#     #     embeddings = embeddings.cuda()
#     if not cluster_centers.is_cuda:
#         # assign to embedding device
#         cluster_centers = cluster_centers.to(embeddings.device)
    
    
#     batch_size, time_steps, embedding_dim = embeddings.shape
#     # num_clusters, _ = cluster_centers.shape

#     # Reshape embeddings to (batch_size * time_steps, embedding_dim)
#     # print(embeddings.shape)
#     embeddings_flat = embeddings.view(-1, embedding_dim)  # Shape: (batch_size * time_steps, embedding_dim)

#     # Compute squared Euclidean distances between embeddings and cluster centers
#     # Using broadcasting to efficiently compute the distance
#     distances = torch.cdist(embeddings_flat, cluster_centers, p=2)  # Shape: (batch_size * time_steps, num_clusters)

#     # Get the index of the nearest cluster for each embedding
#     nearest_clusters = torch.argmin(distances, dim=1)  # Shape: (batch_size * time_steps)
    
#     # Reshape the result back to (batch_size, time_steps, 1)
#     nearest_clusters = nearest_clusters.view(batch_size, time_steps)
#     #convert nearest_clusters to list
#     nearest_clusters =  nearest_clusters.cpu().numpy()#.astype(str)#.tolist()
#     # nearest_clusters = nearest_clusters.cpu().numpy().tolist()
    
#     # map each element to str
#     # nearest_clusters = [[str(token_vocab[i]) for i in j] for j in nearest_clusters]

#     return nearest_clusters




# from collections import defaultdict

# class TrieNode:
#     def __init__(self):
#         self.children = defaultdict(TrieNode)
#         self.token = None  # BPE key for matched sequence

# def build_bpe_trie(bpe_vocab):
#     root = TrieNode()
#     for key, value in bpe_vocab.items():
#         node = root
#         value_list = value.strip().split(' ')
#         for token in value_list:
#             node = node.children[token]
#         node.token = key  # Mark end of the BPE sequence
#     return root

# def convert_token_list_to_bpe(token_list, bpe_trie, seq_len, max_bpe, is_inference,
#                               compress_noise = False, noise_token = None):
#     # Preprocess BPE vocab into a trie for efficient matching
#     # bpe_trie = build_bpe_trie(bpe_vocab)
#     converted_tokens = token_list.copy()
#     n = len(converted_tokens)
    
#     # compressing
#     if compress_noise:
#         compressed_tokens = []
#         compressed_tokens.append(converted_tokens[0])
#         for i in range(1,n):
#             if int(converted_tokens[i]) == noise_token and int(converted_tokens[i-1]) == noise_token:
#                 continue
#             compressed_tokens.append(converted_tokens[i]) 
#         converted_tokens = compressed_tokens
#         n = len(converted_tokens)
    
#     result_tokens = []
#     idx = 0
#     if n ==1:
#         result_tokens.append(converted_tokens[0])
#     while idx < n:
#         node = bpe_trie
#         match_len = 0
#         matched_token = None

#         # Try to find the longest match in the trie
#         for i in range(idx, n):
#             token = str(converted_tokens[i])  # Ensure we treat tokens as strings
#             if token in node.children:
#                 node = node.children[token]
#                 if node.token is not None:
#                     match_len = i - idx + 1
#                     matched_token = node.token
#             else:
#                 break
        
#         if matched_token is not None:
#             # If a match is found, use the BPE token
#             result_tokens.append(matched_token)
#             idx += match_len  # Skip over the matched sequence
#         else:
#             # No match, add the original token
#             result_tokens.append(converted_tokens[idx])
#             idx += 1

#     # Handle inference padding
#     if is_inference:
#         result_len = len(result_tokens)
#         if result_len < seq_len:
#             padding = np.full(seq_len - result_len, int(max_bpe) + 1, dtype=int)
#             result_tokens = np.concatenate((np.array(result_tokens, dtype=int), padding))
#         result_tokens = np.array(result_tokens, dtype=int)

#     return np.array(result_tokens)

# def convert_to_bpe(nearest_tokens, bpe_vocab,bpe_trie, is_inference=False, num_processes=4,
#                    compress_noise = False, noise_token = None):
#     converted_tokens_list = []
#     seq_len = len(nearest_tokens[0])
#     try:
#         max_bpe = max(bpe_vocab.keys())
#     except ValueError:
#         max_bpe = -1

#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         futures = [executor.submit(convert_token_list_to_bpe, token_list, bpe_trie, seq_len, max_bpe, is_inference, compress_noise, noise_token)
#                    for token_list in nearest_tokens]

#         for future in futures:
#             converted_tokens_list.append(future.result())

#     if is_inference:
#         converted_tokens_list = np.array(converted_tokens_list)

#     return converted_tokens_list




# def chunk_list(data, num_chunks):
#     """Split data into approximately equal chunks."""
#     avg = len(data) / float(num_chunks)
#     out = []
#     last = 0.0

#     while last < len(data):
#         out.append(data[int(last):int(last + avg)])
#         last += avg

#     return out

# def process_batch(sub_nearest_tokens, bpe_vocab, bpe_trie, compress_noise, noise_token):
#     """Process a sub-batch of nearest tokens."""
#     # Convert the combinations of tokens if tokens exist in the bpe_vocab
#     bpe_converted_tokens = convert_to_bpe(sub_nearest_tokens, bpe_vocab, bpe_trie,
#                                           compress_noise=compress_noise, noise_token=noise_token)
#     batch_stats = defaultdict(int)
#     for sample in bpe_converted_tokens:
#         for token_num in range(len(sample) - 1):
#             batch_stats[f'{sample[token_num]} {sample[token_num + 1]}'] += 1
#     return batch_stats

# def get_most_freq_pair(val_loader, bpe_vocab, tokenizer_model, data_set_params,vqvae_params, device, 
#                        compress_noise=False, noise_token=None, num_workers=8):
#     stats = defaultdict(int)
#     bpe_trie = build_bpe_trie(bpe_vocab)

#     for i, val_batch in tqdm(enumerate(val_loader)):
#         # if i == 400:
#         #     break
#         X, labels = val_batch
#         X = X.to(device)
#         if len(X.shape) == 4:
#             B,C,F,T = X.shape
#             X = X.view(-1,F,T)
#         elif len(X.shape) == 3:
#             B,C,T = X.shape
#             X = X.view(-1,T).unsqueeze(1)
            
#         _,x_token = tokenizer_model.tokenize(X)
        
        
#         # Split nearest_tokens into num_workers chunks
#         nearest_token_chunks = chunk_list(x_token.cpu().numpy(), num_workers)

#         # Process the sub-batches in parallel after nearest_tokens assignment
#         with ProcessPoolExecutor(max_workers=num_workers) as executor:
#             futures = [executor.submit(process_batch, sub_nearest_tokens, bpe_vocab, bpe_trie,
#                                        compress_noise, noise_token) 
#                        for sub_nearest_tokens in nearest_token_chunks]

#             for future in as_completed(futures):
#                 batch_stats = future.result()
#                 for pair, count in batch_stats.items():
#                     stats[pair] += count

#     # Sort the stats
#     most_freq_pair, most_freq_pair_count = max(stats.items(), key=lambda x: x[1])
    
#     return most_freq_pair, most_freq_pair_count
        

# def learn_bpe(val_loader,bpe_vocab,max_vocab_size,tokenizer_model,data_set_params,vqvae_params,device,
#               compress_noise = False):
#     most_freq_pair,most_freq_pair_count = get_most_freq_pair(val_loader,bpe_vocab,tokenizer_model,data_set_params,vqvae_params,device)
#     print(f'First most frequent pair: {most_freq_pair} with count: {most_freq_pair_count}')
#     most_freq_pair_list = most_freq_pair.split(' ')
#     noise_token = None
#     if compress_noise == True:
#         if most_freq_pair_list[0] == most_freq_pair_list[1]:
#             print('Most frequent pair is the same, assign to noise token')
#             noise_token = int(most_freq_pair_list[0])
#             print(f'Noise token: {noise_token}')
            
#             print('Checking most frequent pair count after compressing noise')
#             most_freq_pair,most_freq_pair_count = get_most_freq_pair(val_loader,bpe_vocab,tokenizer_model,data_set_params,vqvae_params,device,
#                                                                     compress_noise=True, noise_token=noise_token)
#             print(f'Most frequent pair after compressing noise: {most_freq_pair} with count: {most_freq_pair_count}')
#         else: 
#             compress_noise = False
#             print('Most frequent pair is not the same, disabling compress_noise')
        
#     code_book_size = vqvae_params['code_book_size']
#     # add the most frequent pair to the bpe_vocab
#     bpe_vocab[code_book_size] = most_freq_pair
#     print(f'Added {most_freq_pair} to bpe_vocab')
#     print(f'Most frequent pair count: {most_freq_pair_count}')
#     print(f'Current vocab size: {len(bpe_vocab)}')
#     # print(bpe_vocab)
#     while max(bpe_vocab.keys()) < max_vocab_size-1:
#         bpe_vocab = {k: v for k, v in sorted(bpe_vocab.items(), key=lambda item: len(item[1]), reverse=True)}
#         most_freq_pair,most_freq_pair_count = get_most_freq_pair(val_loader,bpe_vocab,tokenizer_model,data_set_params,vqvae_params,device,
#                                                                  compress_noise=compress_noise, noise_token=noise_token)
#         # print (most_freq_pair_count,most_freq_pair)
#         if most_freq_pair_count < 2:
#             print('Most frequent pair count is less than 2, breaking loop')
#             print(f'Final BPE vocab size: {len(bpe_vocab) + code_book_size}')
#             break
        
#         # check if any token in the most frequent pair is already in the bpe_vocab
#         most_freq_pair_list = np.array(most_freq_pair.split(' ')).astype(int)
#         # print(most_freq_pair_list)
#         i = 0
#         while i < len(most_freq_pair_list):
#             if most_freq_pair_list[i] in bpe_vocab:
#                 # print(most_freq_pair_list)
#                 exist_bpe_token_vals = bpe_vocab[int(most_freq_pair_list[i])]
#                 exist_bpe_token_vals = [int(i) for i in exist_bpe_token_vals.split(' ')]
#                 # split and convert to int most_freq_pair_list[i]
                
#                 most_freq_pair_list = np.delete(most_freq_pair_list, i)
#                 most_freq_pair_list = np.insert(most_freq_pair_list, i, exist_bpe_token_vals)
#                 # print(most_freq_pair_list)
#             else:
#                 i += 1
#         most_freq_pair = ' '.join(most_freq_pair_list.astype(str))
        
        
#         # add the most frequent pair to the bpe_vocab
#         bpe_vocab[max(bpe_vocab.keys())+1] = most_freq_pair
#         print(f'Added {most_freq_pair} to bpe_vocab')
#         print(f'Most frequent pair count: {most_freq_pair_count}')
#         print(f'Current vocab size: {len(bpe_vocab)}')
#     return bpe_vocab, noise_token

###########################################################
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class TrieNode:
    def __init__(self):
        self.children = {}
        self.token = None  # BPE key for matched sequence

def build_bpe_trie(bpe_vocab):
    root = TrieNode()
    for key, value in bpe_vocab.items():
        node = root
        value_list = [int(token) for token in value.strip().split(' ')]
        for token in value_list:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.token = key  # Mark end of the BPE sequence
    return root

def convert_token_list_to_bpe(token_list, bpe_trie, seq_len, max_bpe, is_inference,
                              compress_noise=False, noise_token=None):
    converted_tokens = token_list.copy()
    n = len(converted_tokens)

    # Compress noise if required
    if compress_noise:
        compressed_tokens = [converted_tokens[0]]
        for i in range(1, n):
            if (converted_tokens[i] == noise_token and converted_tokens[i - 1] == noise_token):
                continue
            compressed_tokens.append(converted_tokens[i])
        converted_tokens = compressed_tokens
        n = len(converted_tokens)

    result_tokens = []
    idx = 0
    if n == 1:
        result_tokens.append(converted_tokens[0])
    while idx < n:
        node = bpe_trie
        match_len = 0
        matched_token = None

        # Try to find the longest match in the trie
        for i in range(idx, n):
            token = converted_tokens[i]
            if token in node.children:
                node = node.children[token]
                if node.token is not None:
                    match_len = i - idx + 1
                    matched_token = node.token
            else:
                break

        if matched_token is not None:
            # If a match is found, use the BPE token
            result_tokens.append(matched_token)
            idx += match_len  # Skip over the matched sequence
        else:
            # No match, add the original token
            result_tokens.append(converted_tokens[idx])
            idx += 1

    # Handle inference padding
    if is_inference:
        result_len = len(result_tokens)
        if result_len < seq_len:
            padding = np.full(seq_len - result_len, int(max_bpe) + 1, dtype=int)
            result_tokens = np.concatenate((np.array(result_tokens, dtype=int), padding))
        result_tokens = np.array(result_tokens, dtype=int)

    return np.array(result_tokens, dtype=int)

def process_batch(nearest_tokens, bpe_vocab, bpe_trie, compress_noise, noise_token):
    """Process a batch of nearest tokens."""
    seq_len = len(nearest_tokens[0])
    try:
        max_bpe = max(bpe_vocab.keys())
    except ValueError:
        max_bpe = -1

    batch_stats = defaultdict(int)
    for token_list in nearest_tokens:
        converted_tokens = convert_token_list_to_bpe(
            token_list, bpe_trie, seq_len, max_bpe, is_inference=False,
            compress_noise=compress_noise, noise_token=noise_token
        )
        for i in range(len(converted_tokens) - 1):
            if len(converted_tokens) == 2:
                continue 
            pair = f'{converted_tokens[i]} {converted_tokens[i + 1]}'
            batch_stats[pair] += 1
    return batch_stats

def get_most_freq_pair(val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params, device,
                       compress_noise=False, noise_token=None):
    stats = defaultdict(int)
    bpe_trie = build_bpe_trie(bpe_vocab)
    batch_idx = 0
    for val_batch in tqdm(val_loader):
        # if batch_idx == 300:
        #     break
        batch_idx += 1
        
        X, labels = val_batch
        X = X.to(device)
        if len(X.shape) == 4:
            B, C, F, T = X.shape
            X = X.view(-1, F, T)
        elif len(X.shape) == 3:
            B, C, T = X.shape
            X = X.view(-1, T).unsqueeze(1)

        _, x_token = tokenizer_model.tokenize(X)
        x_token = x_token.cpu().numpy()

        batch_stats = process_batch(
            x_token, bpe_vocab, bpe_trie, compress_noise, noise_token
        )
        for pair, count in batch_stats.items():
            stats[pair] += count

    if not stats:
        return None, 0
    # Get the most frequent pair
    most_freq_pair, most_freq_pair_count = max(stats.items(), key=lambda x: x[1])
    return most_freq_pair, most_freq_pair_count

def learn_bpe(val_loader, bpe_vocab, max_vocab_size, tokenizer_model, data_set_params, vqvae_params, device,
              compress_noise=False):
    most_freq_pair, most_freq_pair_count = get_most_freq_pair(
        val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params, device
    )
    if most_freq_pair is None:
        print('No more pairs to merge.')
        return bpe_vocab, None

    print(f'First most frequent pair: {most_freq_pair} with count: {most_freq_pair_count}')
    most_freq_pair_list = most_freq_pair.split(' ')
    noise_token = None
    if compress_noise:
        if most_freq_pair_list[0] == most_freq_pair_list[1]:
            print('Most frequent pair is the same, assigning to noise token')
            noise_token = int(most_freq_pair_list[0])
            print(f'Noise token: {noise_token}')

            print('Checking most frequent pair count after compressing noise')
            most_freq_pair, most_freq_pair_count = get_most_freq_pair(
                val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params,
                device, compress_noise=True, noise_token=noise_token
            )
            print(f'Most frequent pair after compressing noise: {most_freq_pair} with count: {most_freq_pair_count}')
        else:
            compress_noise = False
            print('Most frequent pair is not the same, disabling compress_noise')

    code_book_size = vqvae_params['code_book_size']
    # Add the most frequent pair to the bpe_vocab
    bpe_vocab[code_book_size] = most_freq_pair
    print(f'Added {most_freq_pair} to bpe_vocab')
    print(f'Most frequent pair count: {most_freq_pair_count}')
    print(f'Current vocab size: {len(bpe_vocab)}')

    while max(bpe_vocab.keys()) < max_vocab_size - 1:
        bpe_vocab = dict(sorted(bpe_vocab.items(), key=lambda item: len(item[1]), reverse=True))
        most_freq_pair, most_freq_pair_count = get_most_freq_pair(
            val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params,
            device, compress_noise=compress_noise, noise_token=noise_token
        )
        if most_freq_pair is None or most_freq_pair_count < 2:
            print('Most frequent pair count is less than 2, breaking loop')
            print(f'Final BPE vocab size: {len(bpe_vocab) + code_book_size}')
            break

        # Expand any existing BPE tokens in the most frequent pair
        most_freq_pair_list = [int(token) for token in most_freq_pair.split(' ')]
        i = 0
        while i < len(most_freq_pair_list):
            token = most_freq_pair_list[i]
            if token in bpe_vocab:
                existing_tokens = [int(t) for t in bpe_vocab[token].split(' ')]
                most_freq_pair_list = most_freq_pair_list[:i] + existing_tokens + most_freq_pair_list[i+1:]
                i += len(existing_tokens)
            else:
                i += 1
        most_freq_pair = ' '.join(map(str, most_freq_pair_list))

        # Add the most frequent pair to the bpe_vocab
        bpe_vocab[max(bpe_vocab.keys()) + 1] = most_freq_pair
        print(f'Added {most_freq_pair} to bpe_vocab')
        print(f'Most frequent pair count: {most_freq_pair_count}')
        print(f'Current vocab size: {len(bpe_vocab)}')
        print(bpe_vocab)
    return bpe_vocab, noise_token

#########################################################

# import torch
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm

# class TrieNode:
#     def __init__(self):
#         self.children = {}
#         self.token = None  # BPE key for matched sequence

# def build_bpe_trie(bpe_vocab):
#     root = TrieNode()
#     for key, value in bpe_vocab.items():
#         node = root
#         value_list = [int(token) for token in value.strip().split(' ')]
#         for token in value_list:
#             if token not in node.children:
#                 node.children[token] = TrieNode()
#             node = node.children[token]
#         node.token = key  # Mark end of the BPE sequence
#     return root

# def convert_token_list_to_bpe(token_list, bpe_trie, seq_len, max_bpe, is_inference,
#                               compress_noise=False, noise_token=None):
#     converted_tokens = token_list.copy()
#     n = len(converted_tokens)

#     # Compress noise if required
#     if compress_noise:
#         compressed_tokens = [converted_tokens[0]]
#         for i in range(1, n):
#             if (converted_tokens[i] == noise_token and converted_tokens[i - 1] == noise_token):
#                 continue
#             compressed_tokens.append(converted_tokens[i])
#         converted_tokens = compressed_tokens
#         n = len(converted_tokens)

#     result_tokens = []
#     idx = 0
#     if n == 1:
#         result_tokens.append(converted_tokens[0])
#     while idx < n:
#         node = bpe_trie
#         match_len = 0
#         matched_token = None

#         # Try to find the longest match in the trie
#         for i in range(idx, n):
#             token = converted_tokens[i]
#             if token in node.children:
#                 node = node.children[token]
#                 if node.token is not None:
#                     match_len = i - idx + 1
#                     matched_token = node.token
#             else:
#                 break

#         if matched_token is not None:
#             # If a match is found, use the BPE token
#             result_tokens.append(matched_token)
#             idx += match_len  # Skip over the matched sequence
#         else:
#             # No match, add the original token
#             result_tokens.append(converted_tokens[idx])
#             idx += 1

#     # Handle inference padding
#     if is_inference:
#         result_len = len(result_tokens)
#         if result_len < seq_len:
#             padding = np.full(seq_len - result_len, int(max_bpe) + 1, dtype=int)
#             result_tokens = np.concatenate((np.array(result_tokens, dtype=int), padding))
#         result_tokens = np.array(result_tokens, dtype=int)

#     return np.array(result_tokens, dtype=int)

# def process_batch(nearest_tokens, bpe_vocab, bpe_trie, compress_noise, noise_token):
#     """Process a batch of nearest tokens."""
#     seq_len = len(nearest_tokens[0])
#     try:
#         max_bpe = max(bpe_vocab.keys())
#     except ValueError:
#         max_bpe = -1

#     batch_stats = defaultdict(int)
#     for token_list in nearest_tokens:
#         converted_tokens = convert_token_list_to_bpe(
#             token_list, bpe_trie, seq_len, max_bpe, is_inference=False,
#             compress_noise=compress_noise, noise_token=noise_token
#         )
#         for i in range(len(converted_tokens) - 1):
#             pair = f'{converted_tokens[i]} {converted_tokens[i + 1]}'
#             batch_stats[pair] += 1
#     return batch_stats

# def get_most_freq_pair(val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params, device,
#                        compress_noise=False, noise_token=None):
#     stats = defaultdict(int)
#     bpe_trie = build_bpe_trie(bpe_vocab)

#     for val_batch in tqdm(val_loader):
#         X, labels = val_batch
#         X = X.to(device)
#         if len(X.shape) == 4:
#             B, C, F, T = X.shape
#             X = X.view(-1, F, T)
#         elif len(X.shape) == 3:
#             B, C, T = X.shape
#             X = X.view(-1, T).unsqueeze(1)

#         _, x_token = tokenizer_model.tokenize(X)
#         x_token = x_token.cpu().numpy()

#         batch_stats = process_batch(
#             x_token, bpe_vocab, bpe_trie, compress_noise, noise_token
#         )
#         for pair, count in batch_stats.items():
#             stats[pair] += count

#     if not stats:
#         return None, 0
#     # Get the most frequent pair
#     most_freq_pair, most_freq_pair_count = max(stats.items(), key=lambda x: x[1])
#     return most_freq_pair, most_freq_pair_count

# def learn_bpe(val_loader, bpe_vocab, max_vocab_size, tokenizer_model, data_set_params, vqvae_params, device,
#               compress_noise=False):
#     code_book_size = vqvae_params['code_book_size']
#     # Initialize next_token_id to be greater than any existing token ID
#     if bpe_vocab:
#         next_token_id = max(max(bpe_vocab.keys()), code_book_size) + 1
#     else:
#         next_token_id = code_book_size

#     most_freq_pair, most_freq_pair_count = get_most_freq_pair(
#         val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params, device
#     )
#     if most_freq_pair is None:
#         print('No more pairs to merge.')
#         return bpe_vocab, None

#     print(f'First most frequent pair: {most_freq_pair} with count: {most_freq_pair_count}')
#     most_freq_pair_list = most_freq_pair.split(' ')
#     noise_token = None
#     if compress_noise:
#         if most_freq_pair_list[0] == most_freq_pair_list[1]:
#             print('Most frequent pair is the same, assigning to noise token')
#             noise_token = int(most_freq_pair_list[0])
#             print(f'Noise token: {noise_token}')

#             print('Checking most frequent pair count after compressing noise')
#             most_freq_pair, most_freq_pair_count = get_most_freq_pair(
#                 val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params,
#                 device, compress_noise=True, noise_token=noise_token
#             )
#             print(f'Most frequent pair after compressing noise: {most_freq_pair} with count: {most_freq_pair_count}')
#         else:
#             compress_noise = False
#             print('Most frequent pair is not the same, disabling compress_noise')

#     # Add the most frequent pair to the bpe_vocab with a unique ID
#     bpe_vocab[next_token_id] = most_freq_pair
#     print(f'Added {most_freq_pair} to bpe_vocab with ID {next_token_id}')
#     print(f'Most frequent pair count: {most_freq_pair_count}')
#     print(f'Current vocab size: {len(bpe_vocab)}')
#     next_token_id += 1

#     while next_token_id < max_vocab_size:
#         bpe_vocab = dict(sorted(bpe_vocab.items(), key=lambda item: len(item[1]), reverse=True))
#         most_freq_pair, most_freq_pair_count = get_most_freq_pair(
#             val_loader, bpe_vocab, tokenizer_model, data_set_params, vqvae_params,
#             device, compress_noise=compress_noise, noise_token=noise_token
#         )
#         if most_freq_pair is None or most_freq_pair_count < 2:
#             print('Most frequent pair count is less than 2, breaking loop')
#             print(f'Final BPE vocab size: {len(bpe_vocab)}')
#             break

#         # Expand any existing BPE tokens in the most frequent pair
#         most_freq_pair_list = [int(token) for token in most_freq_pair.split(' ')]
#         i = 0
#         while i < len(most_freq_pair_list):
#             token = most_freq_pair_list[i]
#             if token in bpe_vocab:
#                 existing_tokens = [int(t) for t in bpe_vocab[token].split(' ')]
#                 most_freq_pair_list = most_freq_pair_list[:i] + existing_tokens + most_freq_pair_list[i+1:]
#                 i += len(existing_tokens)
#             else:
#                 i += 1
#         expanded_pair = ' '.join(map(str, most_freq_pair_list))

#         # Add the most frequent pair to the bpe_vocab with a unique ID
#         bpe_vocab[next_token_id] = expanded_pair
#         print(f'Added {expanded_pair} to bpe_vocab with ID {next_token_id}')
#         print(f'Most frequent pair count: {most_freq_pair_count}')
#         print(f'Current vocab size: {len(bpe_vocab)}')
#         print(bpe_vocab)
#         next_token_id += 1
#     return bpe_vocab, noise_token
