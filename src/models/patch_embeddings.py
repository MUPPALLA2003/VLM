import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
   
    def __init__(self,img_size:int,patch_size:int,num_channels:int,embed_dim:int) -> None:

        super().__init__()

        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=num_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        B,C,H,W = x.shape

        if C != self.num_channels:

            raise ValueError(f"Expected {self.num_channels} channels, got {C}")
        
        if H != self.img_size or W != self.img_size:

            raise ValueError(f"Expected input size {self.img_size}x{self.img_size}, got {H}x{W}")

        x = self.proj(x) 
        x = x.flatten(2)
        x = x.transpose(1,2)

        return x