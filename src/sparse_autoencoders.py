import torch
from torch import nn
import einops

class SAE_topk(nn.Module):
    def __init__(self, meta_data:dict):
        super().__init__()

        input_size = meta_data["input_size"]
        hidden_size = meta_data["hidden_size"]
        self.k = meta_data['k']

        self.pre_encode_b = nn.Parameter(torch.randn(hidden_size)*0.1)

        initial_W = torch.randn(hidden_size, input_size) * 0.01
        
        with torch.no_grad():
            self.W = nn.Parameter(initial_W.clone())
            self.WT = nn.Parameter(initial_W.T.clone())

        self.b1 = nn.Parameter(torch.randn(hidden_size)*0.1)  # Bias for encoder
        self.b2 = nn.Parameter(torch.randn(input_size)*0.1)  # Bias for decoder
        self.hidden_activations = None

    def forward(self, x):

        x = x - self.pre_encode_b

        h = torch.topk(torch.matmul(x, self.WT) + self.b1, k=self.k, dim=-1)
        self.hidden_activations = h
        x_hat = einops.einsum(h.values, self.W[h.indices], 'token topk, token topk out -> token out') + self.b2

        return x_hat

    def get_activations(self, x):

        x = x - self.pre_encode_b

        h = torch.topk(torch.matmul(x, self.WT) + self.b1, k=self.k, dim=-1)
        self.hidden_activations = h

        return h
    
    def get_preacts(self, x):

        x = x - self.pre_encode_b

        return torch.matmul(x, self.WT) + self.b1