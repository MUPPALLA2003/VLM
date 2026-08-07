import torch
import torch.nn as nn
from .vision_encoder_layer import VisionEncoderLayer

class VisionEncoder(nn.Module):

    def __init__(self,num_layers:int,embed_dim:int,n_heads:int,mlp_ratio:float,qkv_bias:bool = True,attn_p:float = 0.0,proj_p:float = 0.0,flash_attn:bool = True,mlp_p:float = 0.0,drop_path:float=0.0) -> None:

        super().__init__()

        self.layers = nn.ModuleList([
            VisionEncoderLayer(embed_dim,n_heads,mlp_ratio,qkv_bias,attn_p,proj_p,flash_attn,mlp_p,drop_path)
            for _ in range(num_layers)
        ])

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        for layer in self.layers:

            x = layer(x)

        return x        