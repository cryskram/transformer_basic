import torch
import torch.nn as nn

vocab_size = 65
embedding_dim = 64

token_embedding_table = nn.Embedding(vocab_size, embedding_dim)

idx = torch.tensor([[1, 2, 3]])
x = token_embedding_table(idx)
print(x.shape)

BLOCK_SIZE = 128

position_embedding_table = nn.Embedding(BLOCK_SIZE, embedding_dim)

T = idx.shape[1]
pos = torch.arange(T)

pos_embedding = position_embedding_table(pos)

token_embedding = token_embedding_table(idx)
x = token_embedding + pos_embedding
print(x.shape)
