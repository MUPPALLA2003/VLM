import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from .rope import precompute_rope_params,apply_rope

class Attention(nn.Module):

    def __init__(self,embed_dim:int,max_seq_len:int,num_heads:int,n_kv_heads:Optional[int] = None,attn_p:float=0.0,bias:bool = False,rope_base:float = 500000.0,flash_attn:bool = True) -> None:

        super().__init__()

        assert embed_dim % num_heads == 0

        n_kv_heads = n_kv_heads or num_heads

        assert num_heads % n_kv_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.n_rep = num_heads // n_kv_heads
        self.head_dim = embed_dim // num_heads
        self.flash_attn = flash_attn
        self.attn_drop = nn.Dropout(attn_p)

        self.wq = nn.Linear(embed_dim,num_heads * self.head_dim,bias = bias)
        self.wk = nn.Linear(embed_dim,n_kv_heads * self.head_dim,bias = bias)
        self.wv = nn.Linear(embed_dim,n_kv_heads * self.head_dim,bias = bias)
        self.wo = nn.Linear(embed_dim,embed_dim,bias = bias)

        causal_mask = torch.triu(torch.ones(max_seq_len,max_seq_len,dtype = bool),diagonal=1)
        self.register_buffer("causal_mask",causal_mask,persistent=False)

        cos,sin = precompute_rope_params(self.head_dim,max_seq_len,rope_base)
        self.register_buffer("cos_cache",cos,persistent=False) 
        self.register_buffer("sin_cache",sin,persistent=False)

    def _repeat_kv(self,x:torch.Tensor,n_rep:int) -> torch.Tensor:
  
        if n_rep == 1:

            return x
        
        batch,n_kv_heads,seq_len,head_dim = x.shape
        grouped_x = (x[:,:,None,:,:].expand(batch,n_kv_heads,n_rep,seq_len,head_dim).reshape(batch,n_kv_heads * n_rep,seq_len,head_dim))

        return grouped_x

    def _attention(self,query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask:Optional[torch.Tensor]) -> torch.Tensor:

        scores = query @ key.transpose(-2,-1) / math.sqrt(self.head_dim)

        if mask is not None:

            scores = scores.masked_fill(mask,float("-inf"))

        output = F.softmax(scores,dim=-1,dtype=torch.float32).to(scores.dtype) @ value

        return output

    def forward(self,x:torch.Tensor,kv_cache = None,layer_idx = None) -> torch.Tensor:

        batch_size,seq_len,C = x.shape

        assert seq_len <= self.max_seq_len

        assert C == self.embed_dim 

        query = self.wq(x).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        key = self.wk(x).view(batch_size,seq_len,self.n_kv_heads,self.head_dim).transpose(1,2)
        value = self.wv(x).view(batch_size,seq_len,self.n_kv_heads,self.head_dim).transpose(1,2)

        if kv_cache is None:

            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
            query = apply_rope(query,cos,sin)
            key = apply_rope(key,cos,sin)     
            key = self._repeat_kv(key,self.n_rep)
            value = self._repeat_kv(value,self.n_rep)

            if self.flash_attn:

                output = F.scaled_dot_product_attention(query,key,value,
                        attn_mask=None,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                        is_causal=True)

            else:

                mask = self.causal_mask[:seq_len,:seq_len]
                output = self._attention(query,key,value,mask)
                
        else:

            assert layer_idx is not None, "layer_idx required with kv_cache"

            assert (start_pos + seq_len <= self.max_seq_len), "KV cache exceeded max_seq_len"
 
            start_pos = kv_cache.seq_len[layer_idx]
            cos = self.cos_cache[start_pos : start_pos + seq_len]
            sin = self.sin_cache[start_pos : start_pos + seq_len]
            q = apply_rope(query,cos,sin)
            k = apply_rope(key,cos,sin)     
            k, v, _ = kv_cache.update(layer_idx,k,v)
            k = self._repeat_kv(k,self.n_rep)
            v = self._repeat_kv(v,self.n_rep)
            s_q  = q.shape[-2]
            s_kv = k.shape[-2]
            offset = s_kv - s_q

            mask = (torch.arange(s_kv,device=x.device)[None,:] > torch.arange(s_q, device=x.device)[:, None] + offset)
            
            output = self._attention(q,k,v,mask)
 
        output = output.transpose(1, 2).contiguous().view(batch_size,seq_len,self.embed_dim)

        return self.wo(output)