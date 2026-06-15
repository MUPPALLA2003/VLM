import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization

class ResidualNetwork(nn.Module):

    def _init_(self,d_model:int,dropout:float):

        super()._init_()
        
        self.n_embd = d_model
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNormalization(d_model)


    def forward(self,x:torch.Tensor,sublayer) -> torch.Tensor:

        return x + self.dropout(sublayer(LayerNormalization(x)))