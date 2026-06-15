import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self,d_model:int,eps:float=1e-5,device:torch.device | None = None):

        super().__init__()
        self.n_dim = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model, device = device, dtype = torch.float32))
        self.beta = nn.Parameter(torch.zeros(d_model, device = device, dtype = torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.shape[-1] == self.n_dim

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)

        return self.gamma * x_hat + self.beta