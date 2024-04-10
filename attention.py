import torch
import torch.nn.functional as F
import math

class Attention(torch.nn.Module):
    def __init__(self, Q_dim, K_dim, head_size):
        """
        head_size: it will be 256 but can be changed and its being used to project into same feature dimentions of q, k , and v.
        """
        super().__init__()
        self.query = torch.nn.Linear(Q_dim, head_size, bias=False)
        self.key   = torch.nn.Linear(K_dim, head_size, bias=False)
        self.value = torch.nn.Linear(K_dim, head_size, bias=False)
        
    def forward(self, Q_in, K_in, attention_mask):
        q = self.query(Q_in) 
        k = self.key(K_in) # (B,T,Head size) same for all q, k, v
        v = self.value(K_in)

        print(f"k shape : {k.shape} | q shape: {q.shape}")

        wei = q @ k.transpose(1,2)  * (1 / math.sqrt(q.size(-1)))  # (B, T, 256) @ (B, 256, T)   ---> (B, T, T)
        print("wei shape: ", wei.shape)
        # wei = torch.zeros((T,T))
        wei = wei.masked_fill(attention_mask == 0, float('-inf'))
        # wei = wei.masked_fill(attention_mask[:,:50,:50] == 0, float('-inf'))
        # wei=wei.masked_fill(~attention_mask,-1e10)
        wei = F.softmax(wei, dim=-1) # attention map
        
        out = wei @ v # (B, T, T) * (B, T, 256) ---> (B, T, 256)
        return out, wei