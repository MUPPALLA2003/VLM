import torch
import torch.nn as nn

class LMHead(nn.Module):

    def __init__(self,embed_dim:int,vocab_size:int) -> None:

        super().__init__()

        self.lm_head = nn.Linear(embed_dim,vocab_size,bias = False)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        x = self.lm_head(x)

        return x    