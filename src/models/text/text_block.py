import torch
import torch.nn as nn
from typing import Optional
from .kv_cache import KVCache
from .decoder import TextDecoder
from .text_embeddings import TextEmbeddings
from .rms_norm import RMSNormalization

class TextBlock(nn.Module):

    def __init__(self,
            num_layers:int,
            vocab_size:int,
            intermediate_dim:int,
            embed_dim:int,
            max_seq_len:int,
            num_heads:int,
            n_kv_heads:Optional[int] = None,
            attn_p:float=0.0,
            attn_bias:bool = False,
            mlp_bias:bool = False,
            rope_base:float = 500000.0,
            flash_attn:bool = True,
            ) -> None:

        super().__init__()

        self.input_embed = TextEmbeddings(vocab_size,embed_dim)

        self.layers = nn.ModuleList([TextDecoder(
            intermediate_dim,
            embed_dim,
            max_seq_len,
            num_heads,
            n_kv_heads,
            attn_p,
            attn_bias,
            mlp_bias,
            rope_base,
            flash_attn
        )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNormalization(embed_dim)  

    def forward(self,input_ids:torch.Tensor,kv_cache:Optional[KVCache] = None) -> torch.Tensor:

        x = self.input_embed(input_ids)

        for layer_idx,layer in enumerate(self.layers):

            x = layer(x,kv_cache,layer_idx)

        x = self.final_norm(x)

        return x     

