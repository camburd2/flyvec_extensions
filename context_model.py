import torch.nn as nn
import time
import torch


class ContextModel():
    def __init__(self, K_size, vocab_size, k, lr, norm_rate=0, device='cpu'):
        self.k = k
        self.lr = lr
        self.W = torch.empty(K_size, vocab_size, device=device).uniform_(0, .1)

        self.norm_rate = norm_rate
        self.count = 0
        self.norm_mask = torch.zeros(K_size, dtype=bool, device=device)


    def forward(self, x):
        active_indices = x.nonzero().squeeze(1)
        return torch.sum(self.W[:,active_indices], dim=1)


    def update(self, x):
        # Find indices of words in input (onehot vocab)
        active_indices = x.nonzero().squeeze(1)

        # Calculate top k activations
        activations = torch.sum(self.W[:,active_indices], dim=1)
        topk_vals, topk_idx = torch.topk(activations, self.k)

        # Update weights from active inputs (words) to active neurons (topk)
        updates = self.lr * topk_vals[:, None]
        self.W[topk_idx[:, None], active_indices] += updates

        # Normalize updated rows (incoming weights to neurons with modified weights since last normalization)
        self.norm_mask[topk_idx] = True
        self.count += 1
        if self.count > self.norm_rate:
            self.count = 0
            self.W[self.norm_mask] = nn.functional.normalize(self.W[self.norm_mask], dim=1)
            self.norm_mask.zero_()


    def get_embedding(self, x, hash_len):
        # Get top k activations
        activations = self.forward(x)
        _, top_k_indices = torch.topk(activations, hash_len)
        
        # Create word embedding
        embedding = torch.zeros_like(activations)
        embedding[top_k_indices] = 1
        
        return embedding
    
