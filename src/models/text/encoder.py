import torch
import torch.nn as nn
from typing import Optional
from .attention import Attention
from .gated_mlp import  GatedMLP
from .rms_norm import RMSNormalization
from .kv_cache import KVCache

class TextEncoder(nn.Module):

    def __init__(self,intermediate_dim:int,embed_dim:int,max_seq_len:int,num_heads:int,n_kv_heads:Optional[int] = None,attn_p:float=0.0,attn_bias:bool = False,mlp_bias:bool = False,rope_base:float = 500000.0,flash_attn:bool = True) -> None:

        self.attention = Attention(embed_dim,max_seq_len,num_heads,n_kv_heads,attn_p,attn_bias,rope_base,flash_attn)
        self.mlp = GatedMLP(embed_dim,intermediate_dim,mlp_bias)
        self.attn_norm = RMSNormalization(embed_dim)
        self.mlp_norm = RMSNormalization(embed_dim)

    def forward(self,x:torch.Tensor,kv_cache:Optional[KVCache] = None,layer_idx:Optional[int] = None) -> torch.Tensor:

        x = x + self.attention(self.attn_norm(x),kv_cache,layer_idx)
        x = x + self.mlp(self.mlp_norm(x))

        return x