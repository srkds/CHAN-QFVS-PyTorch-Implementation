import torch
import torch.nn.functional as F
import math

class Attention(torch.nn.Module):
  def __init__(self, Q_dim, K_dim, head_size):
    super().__init__()
    self.query = torch.nn.Linear(in_features=Q_dim, out_features=head_size, bias=False)
    self.key = torch.nn.Linear(in_features=K_dim, out_features=head_size, bias=False)
    self.value = torch.nn.Linear(in_features=K_dim, out_features=head_size, bias=False)

  def forward(self, Q_in, K_in, attention_mask):
    """
    input dimention: (B, T, C) -> batch, temporal/sequence, channel
    """
    q = self.query(Q_in)
    k = self.key(K_in)
    v = self.value(K_in)

    wei = q @ k.transpose(1,2) * (1 / math.sqrt(q.size(-1))) # -> (B,T,C) @ (B, C, T) --> (B, T, T)
    wei = wei.masked_fill(attention_mask==0, float('-inf')) # mask the attn map
    wei = F.softmax(wei, dim=-1) # attention map

    out = wei @ v # (B,T,T) @ (B, T, C)(B, T, C)
    return out, wei

if __name__ == "__main__":
    # self Attention
    in_seq = torch.rand((1,20,100)) # B, T, C -> 1, 20, 100
    attn_mask = torch.zeros(20)
    attn_mask[0:10] = 1

    self_attn = Attention(100, 100, 50)
    attn_result,_ = self_attn(in_seq, in_seq, attn_mask)
    print(attn_result.shape)  # 1 , 20 ,50
