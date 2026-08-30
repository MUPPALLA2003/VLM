import torch
import torch.nn as nn

class RMSNormalization(nn.Module):

    def __init__(self,embed_dim:int,eps:float=1e-05) -> None:

        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim),dtype = torch.float32)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        t = x.float()
        inv = torch.rsqrt(torch.mean(t**2,dim=-1,keepdim = True) + self.eps)

        return (t*inv*self.scale).to(x.dtype)    