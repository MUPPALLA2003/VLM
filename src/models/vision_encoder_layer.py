import torch
import torch.nn as nn
from .VisionAttention import VisionAttention
from .VisionMLP import VisionMLP
from .LayerNormalization import LayerNormalization

class VisionEncoderLayer(nn.Module):

    def __init__(self,embed_dim:int,n_heads:int,mlp_ratio:float,qkv_bias:bool = True,attn_p:float = 0.0,proj_p:float = 0.0,flash_attn:bool = True,mlp_p:float = 0.0,drop_path:float=0.0) -> None:

        super().__init__()

        self.attention = VisionAttention(embed_dim,n_heads,qkv_bias,attn_p,proj_p,flash_attn)
        self.mlp = VisionMLP(embed_dim,mlp_ratio,mlp_p)
        self.layernorm1 = LayerNormalization(embed_dim)
        self.layernorm2 = LayerNormalization(embed_dim)
        self.drop_path1 = nn.Dropout(drop_path)
        self.drop_path2 = nn.Dropout(drop_path)
       
    def forward(self,x:torch.Tensor) -> torch.Tensor:

        x = x + self.drop_path1(self.attention(self.layernorm1(x)))
        x = x + self.drop_path2(self.mlp(self.layernorm2(x)))

        return x   
