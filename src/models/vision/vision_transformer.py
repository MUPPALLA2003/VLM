import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder
from .patch_embeddings import PatchEmbeddings
from .layer_normalization import LayerNormalization

class VisionBlock(nn.Module):

    def __init__(self,
            out_dim:int,
            img_size:int,
            patch_size:int,
            num_channels:int,
            num_layers:int,
            embed_dim:int,
            n_heads:int,
            mlp_ratio:float,
            qkv_bias:bool = True,
            attn_p:float = 0.0,
            proj_p:float = 0.0,
            flash_attn:bool = True,
            mlp_p:float = 0.0,
            drop_path:float = 0.0,
            pos_p:float = 0.0,
            custom_weights:bool = True,
        ) -> None:

        super().__init__()

        if embed_dim % n_heads != 0:
           
           raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")
        
        if img_size % patch_size != 0:

           raise ValueError(f"image_size ({img_size}) must be divisible by patch_size ({patch_size})")

        self.vision_embed = PatchEmbeddings(img_size,patch_size,num_channels,embed_dim)
        num_tokens = self.vision_embed.n_patches + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

        self.vison_encoder = VisionEncoder(num_layers
                                           ,embed_dim,
                                           n_heads,
                                           mlp_ratio,
                                           qkv_bias,
                                           attn_p,
                                           proj_p,
                                           flash_attn,
                                           mlp_p,drop_path)

        self.norm = LayerNormalization(embed_dim)
        self.proj_head = nn.Linear(embed_dim,out_dim)
        self.pos_drop = nn.Dropout(pos_p)

        if custom_weights:

            self.apply(self._init_weights)
         
    def _cls_pos_embed(self,x:torch.Tensor) -> torch.Tensor:

        x = torch.cat([self.cls_token.expand(x.shape[0],-1,-1),x],dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        return x
    
    def _init_weights(self,module:nn.Module):

        if isinstance(module,VisionBlock):

            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data,mean=0,std=0.02)
            module.pos_embed.data = nn.init.trunc_normal_(module.pos_embed.data,mean=0,std=0.02)

        elif isinstance(module,(nn.Linear,nn.Conv2d)):
    
            module.weight.data = nn.init.trunc_normal_(module.weight.data,mean=0,std=0.02)
    
            if module.bias is not None:

                module.bias.data.zero_()

        elif isinstance(module, LayerNormalization):
    
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)

    def forward(self,input_ids:torch.Tensor) -> torch.Tensor:

        if input_ids.dim() != 4:

            raise ValueError(f"Expected input (B,C,H,W), got shape {input_ids.shape}")

        x = self.vision_embed(input_ids)
        x = self._cls_pos_embed(x)
        x = self.vison_encoder(x)
        x = self.norm(x)
        x = x[:,0] 
        x = self.proj_head(x)

        return x