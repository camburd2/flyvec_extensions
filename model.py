import torch

class FlyvecModel:
    def __init__(self, K_size, vocab_size, k, lr, norm_rate=0, device='cpu'):
        self.k = k
        self.lr = lr
        self.W = torch.empty(K_size, vocab_size, device=device).uniform_(0, .1)

        self.norm_rate = norm_rate
        self.norm_count = 0
        self.norm_mask = torch.zeros(K_size, dtype=bool, device=device)


    def _forward(self, x):
        """Compute activations for input x
        
        Find indices of words in input vector (onehot vocab vector).
        Inputs are are binary so only need to sum cols (across k neurons) instead of multiply.
        """        
        active_indices = x.nonzero().squeeze(1)
        activations = torch.sum(self.W[:, active_indices], dim=1)
        return active_indices, activations
    

    def update(self, x):
        """Update model weights based on input x"""
        active_indices, activations = self._forward(x)
        
        # Get top-k neuron indices
        topk_vals, topk_idx = torch.topk(activations, self.k)

        # Update weights from active inputs (words) to active neurons (top-k)
        self.W[topk_idx[:, None], active_indices] += self.lr * topk_vals[:, None]
     
        # Track neurons that were updated for normalization
        self.norm_mask[topk_idx] = True
        self.norm_count += 1

        # Perform normalization 
        if self.norm_count > self.norm_rate:
            # Normalize updated rows (incoming weights to neurons w/ modified weights since last normalization)
            self.W[self.norm_mask] = torch.nn.functional.normalize(self.W[self.norm_mask], dim=1)
            self.norm_mask.zero_()
            self.norm_count = 0  


    def get_embedding(self, x, hash_len):
        """Get sparse word embedding for input x"""      
        _, activations = self._forward(x)
        _, top_k_indices = torch.topk(activations, hash_len)

        # Create word embedding
        embedding = torch.zeros_like(activations)
        embedding[top_k_indices] = 1

        return embedding
