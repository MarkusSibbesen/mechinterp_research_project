import torch
from torch import nn
import einops

class SAE(nn.Module):

    def __init__(self, meta_data:dict):
        super().__init__()

        #meta_data dict needs to have input_size, hidden_size, k, pre_encoder_bias, activation_function

        input_size = meta_data["input_size"]
        hidden_size = meta_data["hidden_size"]
        self.meta_data = meta_data

        initial_W = torch.randn(hidden_size, input_size) * 0.01
        with torch.no_grad():
            self.W = nn.Parameter(initial_W.clone())
            if not self.meta_data['same_W']:
                self.WT = nn.Parameter(initial_W.T.clone())
        

        if meta_data["pre_encoder_bias"]:
            self.pre_encode_b = nn.Parameter(torch.randn(input_size)*0.1)
        self.b1 = nn.Parameter(torch.randn(hidden_size)*0.1)  # Bias for encoder

        self.activations = None

    def get_preacts(self, x):
        if self.meta_data["pre_encoder_bias"]:
            x = x - self.pre_encode_b

        if self.meta_data['same_W']:
            return torch.matmul(x, self.W.t()) + self.b1
        else:
            return torch.matmul(x, self.WT) + self.b1



class SAE_topk(SAE):

    def forward(self, x):
        if self.meta_data["pre_encoder_bias"]:
            x = x - self.pre_encode_b

        if self.meta_data['same_W']:
            h = torch.topk(torch.matmul(x, self.W.t()) + self.b1, k=self.meta_data['k'], dim=-1)
        else:
            h = torch.topk(torch.matmul(x, self.WT) + self.b1, k=self.meta_data['k'], dim=-1)
        self.activations = h
        self.active_neurons = len(torch.unique(h.indices))
        x_hat = einops.einsum(h.values, self.W[h.indices], 'token topk, token topk out -> token out')

        if self.meta_data['pre_encoder_bias']:
            x_hat += self.pre_encode_b

        return x_hat

    def get_activations(self, x):
        if self.meta_data['same_W']:
            h = torch.topk(torch.matmul(x, self.W.t()) + self.b1, k=self.meta_data['k'], dim=-1)
        else:
            h = torch.topk(torch.matmul(x, self.WT) + self.b1, k=self.meta_data['k'], dim=-1)
        return h
    

    

class SAE_RELU(SAE):

    def forward(self, x):
        if self.meta_data["pre_encoder_bias"]:
            x = x - self.pre_encode_b

        if self.meta_data['same_W']:
            h = torch.relu(torch.matmul(x, self.W.t()) + self.b1)
        else:
            h = torch.relu(torch.matmul(x, self.WT) + self.b1)

        self.activations = h
        self.active_neurons_per_batch = sum(h.sum(dim=0) > 0.001)
        #self.active_neurons = torch.sum(h > 0).item()

        x_hat = torch.matmul(h, self.W)
        
        if self.meta_data['pre_encoder_bias']:
            x_hat += self.pre_encode_b

        return x_hat
    
    def get_activations(self, x):
        if self.meta_data['same_W']:
            h = torch.relu(torch.matmul(x, self.W.t()) + self.b1)
        else:
            h = torch.relu(torch.matmul(x, self.WT) + self.b1)
        return h

