import torch
from torch import nn
import einops

class SAE_topk(nn.Module):
    def __init__(self, meta_data:dict):
        super().__init__()

        #meta_data dict needs to have input_size, hidden_size, k, pre_encoder_bias, activation_function

        input_size = meta_data["input_size"]
        hidden_size = meta_data["hidden_size"]
        self.k = meta_data['k']
        self.meta_data = meta_data

        self.pre_encode_b = nn.Parameter(torch.randn(hidden_size)*0.1)

        initial_W = torch.randn(hidden_size, input_size) * 0.01
        
        with torch.no_grad():
            self.W = nn.Parameter(initial_W.clone())
            self.WT = nn.Parameter(initial_W.T.clone())


        self.b1 = nn.Parameter(torch.randn(hidden_size)*0.1)  # Bias for encoder
        self.b2 = nn.Parameter(torch.randn(input_size)*0.1)  # Bias for decoder
        self.hidden_activations = None

    def forward(self, x):
        if self.meta_data["pre_encoder_bias"]:
            x = x - self.b2

        if self.meta_data["activation_function"] == "topk":
            h = torch.topk(torch.matmul(x, self.WT) + self.b1, k=self.k, dim=-1)
            self.hidden_activations = h
            self.active_neurons = len(torch.unique(hiddens.indices))
            x_hat = einops.einsum(h.values, self.W[h.indices], 'token topk, token topk out -> token out') + self.b2
        else:
            h = torch.relu(torch.matmul(x, self.WT) + self.b1)
            self.hidden_activations = h 
            

#            acts = sae_trainers[0].model.hidden_activations
            self.active_neurons = sum(h.sum(dim=0) > 0.001)
            #self.active_neurons = torch.sum(h > 0).item()

            x_hat = torch.matmul(h, self.W) + self.b2

        return x_hat

    def get_activations(self, x):
        
        if self.meta_data["activation_function"] == "topk":
            h = torch.topk(torch.matmul(x, self.WT) + self.b1, k=self.k, dim=-1)
        else:
            h = torch.relu(torch.matmul(x, self.WT) + self.b1)

        return h
    
    def get_preacts(self, x):
        if self.meta_data["pre_encoder_bias"]:
            x = x - self.b2

        return torch.matmul(x, self.WT) + self.b1