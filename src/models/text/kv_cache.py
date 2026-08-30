import torch

class KVCache:

    def __init__(self,num_layers:int,max_batch_size:int,num_heads:int,max_Seq_len:int,head_dim:int,device:str) -> None:

        shape = (num_layers,max_batch_size,num_heads,max_Seq_len,head_dim)

        self.key_cache = torch.zeros(shape,dtype = torch.float16,device = device)
        self.value_cache = torch.zeros(shape,dtype = torch.float16,device = device)
        self.index = torch.zeros(num_layers,dtype = torch.long,device = device)

    def update_cache(self,layer_idx:int,keys:torch.Tensor,values:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:

        batch_size,_,seq_len,_ = keys.shape
        start = self.index[layer_idx]
        end = self.start + seq_len

        self.key_cache[layer_idx,:batch_size,:,start:end,:] = keys
        self.value_cache[layer_idx,:batch_size,:,start:end,:] = values
        self.index[layer_idx] = end

        cached_keys = self.key_cache[layer_idx,:batch_size,:,:end,:]
        cached_values = self.value_cache[layer_idx,:batch_size,:,:end,:]  

        return cached_keys,cached_values

    def reset(self):

        self.index.zero_() 

        