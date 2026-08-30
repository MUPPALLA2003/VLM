import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMLP(nn.Module):

    def __init__(self,hidden_dim:int,intermediate_dim:int,bias:bool = False):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_dim,intermediate_dim,bias=bias)
        self.up_proj = nn.Linear(hidden_dim,intermediate_dim,bias=bias)
        self.down_proj = nn.Linear(intermediate_dim,hidden_dim,bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        
        return self.down_proj(gate * up)  

        