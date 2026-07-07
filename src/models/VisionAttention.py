import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionAttention(nn.Module):

    def __init__(self,embed_dim:int,n_heads:int,qkv_bias:bool = True,attn_p:float = 0.0,proj_p:float = 0.0,flash_attn:bool = True) -> None:

        super().__init__()

        if embed_dim % n_heads != 0:

            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.Q = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.K = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.V = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)
        self.flash_attn = flash_attn

    def attention(self,q,k,v):

        if self.flash_attn:

            y = F.scaled_dot_product_attention(q,k,v,dropout_p = self.attn_drop if self.training else 0)

        else:

            attention_logits = (q @ k.transpose(-2,-1)) * self.scale
            attention_weights = F.softmax(attention_logits,dim = -1)
            attention_weights = self.attn_drop(attention_weights)
            out = attention_weights @ v
            
        return out   

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        batch_size,seq_len,C = x.shape

        if C != self.embed_dim:

            raise ValueError(f"Expected last dim {self.embed_dim}, got {C}")
        
        query = self.Q(x).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        key = self.K(x).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        value = self.V(x).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        output = self.attention(query,key,value)
        output = output.transpose(1,2).contiguous().view(batch_size,seq_len,C)
        output = self.proj(output)
        output = self.proj_drop(output)
                      
        return output 