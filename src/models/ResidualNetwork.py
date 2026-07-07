import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization

class ResidualNetwork(nn.Module):

    def _init_(self,embed_dim:int,residual_dropout:float):

        super()._init_()
        
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.layernorm = LayerNormalization(embed_dim)


    def forward(self,x:torch.Tensor,sublayer) -> torch.Tensor:

        return x + self.residual_dropout(sublayer(LayerNormalization(x)))