import torch
import torch.nn as nn

class TextEmbeddings(nn.Module):

    def __init__(self,vocab_size:int,embed_dim:int) -> None:

        super().__init__()

        self.text_embed = nn.Embedding(vocab_size,embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self,ids:torch.Tensor) -> torch.Tensor:

        x = self.text_embed(ids) * self.scale
        
        return x   