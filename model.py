import torch
import math

from attention import Attention
from utils import load_json

class CHAN(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.conv1d_1 = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=self.config["in_channel"], out_channels=self.config["conv1_channel"], kernel_size=5,stride=1,padding=2),
        torch.nn.BatchNorm1d(self.config["conv1_channel"]),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool1d(2, stride=2, padding=0)
    )

    self.conv1d_2 = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=self.config["conv1_channel"], out_channels=self.config["conv2_channel"], kernel_size=5,stride=1,padding=2),
        torch.nn.BatchNorm1d(self.config["conv2_channel"]),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool1d(2, stride=2, padding=0)
    )

    self.self_attention = Attention(self.config["conv2_channel"], self.config["conv2_channel"], self.config["conv2_channel"])
    self.segment_level_attention = Attention(self.config["concept_dim"], self.config["conv2_channel"], self.config["conv2_channel"])
    self.global_attention = Attention(self.config["conv2_channel"], self.config["conv2_channel"], self.config["conv2_channel"])

    self.de_conv1d_1 = torch.nn.ConvTranspose1d(3*self.config["conv2_channel"], self.config["deconv1_channel"],kernel_size=4,stride=2,padding=1)
    self.de_conv1d_2 = torch.nn.ConvTranspose1d(self.config["deconv1_channel"],self.config["deconv2_channel"],kernel_size=4,stride=2,padding=1)

    self.vision_projector = torch.nn.Linear(self.config["deconv2_channel"],self.config["similarity_dim"],bias=False)
    self.textual_projector = torch.nn.Linear(self.config["concept_dim"],self.config["similarity_dim"],bias=False)

    self.MLP = torch.nn.Linear(self.config["similarity_dim"],1)

  def forward(self, visual_features, seg_len, query1_emd, query2_emd):
    B, seg, T, C = visual_features.shape

    # (B, seg, T, C) -> (B*seg, T, C) -> (B*seg, C, T) Convolutional layer takes (batch, channel, temporal sequence) as ip dimention
    vis1 = self.conv1d_1(visual_features.view(B*seg, T, C).transpose(1,2))
    vis2 = self.conv1d_2(vis1)

    vis2 = vis2.transpose(1,2) # (B*seg, C, T) -> (B*seg, T, C)

    attention_mask = torch.zeros(B, seg, T//4, dtype=torch.bool) # (B, seg, T//4)
    for i in range(B):
      for j in range(len(seg_len[i])):
        for k in range(int(math.ceil(seg_len[i][j]//4.0))):
          attention_mask[i][j][k] = 1

    attention_mask = attention_mask.view(B*seg, -1).unsqueeze(1) # (B, seg, T//4) -> (B*seg, 1, T//4)

    # (B*seg, C, T/4) -> (B*seg, T/4, C) again back at original dimention
    q_mat = vis2.view(B*seg, int(T//4), -1)
    k_mat = vis2.view(B*seg, int(T//4), -1)

    #### self-attention
    self_attention_result, sa_wei = self.self_attention(q_mat, k_mat, attention_mask)

    #### segment-level query conditined attention

    avg_query = (query1_emd + query2_emd) / 2 # average representation of the both query (B, C) -> (B, 300)

    # as avg_query has dimentions (Batch, Channel) -> (batch_size, 300)
    # we want here condition quey for each timestamp of video
    # so make avg_query for each temporal dimention
    #  (batch, T/4, C) -> (batch, 50, 300)
    # 50 is temporal dimention and each timestamp contains same features of 300 dimentions
    # transision (B, 300) -> (B, 1, 1, 300) -> (B, seg, T/4, 300) -> (B*seg, T/4, 300)
    t_q = avg_query.unsqueeze(1).unsqueeze(1).expand(B, seg, int(T/4), 300).contiguous().view(B*seg, int(T/4),300)

    segment_level_result, sl_wei = self.segment_level_attention(t_q, k_mat, attention_mask) # (B*seg, T/4, 256)

    #  (B*seg, T/4, 256) (200, 50, 256) ---> (B*seg, C) -> (200, 256) which is segment wise query relevence score, 1 vector per segment
    segment_level_agg_result = segment_level_result.mean(dim=1)
    segment_level_agg_result = segment_level_agg_result.unsqueeze(0).expand(T//4, B*seg, 256) # (B*seg, 256) -> (T/4, B*seg, 256)
    q_mat = q_mat.contiguous().view(T//4, B*seg, 256) # (B*seg, T/4, 256) --> (T/4, B*seg, 256) == (50, 200, 256)

    #### query-aware global attention
    attention_mask = attention_mask.view(T//4, 1, B*seg)  # (B*seg, 1, T/4) --> (T/4, 1, B*seg)
    global_attention_result, global_wei = self.global_attention(q_mat, segment_level_agg_result, attention_mask) # (T/4, B*seg, 256)
    global_attention_result = global_attention_result.view(B*seg, T//4, -1) # (T/4, B*seg, 256) -> (B*seg, T/4, 256)

    # concate self-attn, global-attn, and conv features
    concat_result = torch.cat((vis2, self_attention_result, global_attention_result), dim=-1) # (B*seg, T/4, 768)

    deconv_result = self.de_conv1d_1(concat_result.transpose(1,2))
    deconv_result = self.de_conv1d_2(deconv_result).transpose(1,2).contiguous().view(B, seg*T, -1)

    visual_similarity =  self.vision_projector(deconv_result)

    query1_similarity = self.textual_projector(query1_emd)
    query2_similarity = self.textual_projector(query2_emd)

    query1_similarity = query1_similarity.unsqueeze(1)*visual_similarity
    query2_similarity = query2_similarity.unsqueeze(1)*visual_similarity

    query1_logit = self.MLP(query1_similarity)
    query2_logit = self.MLP(query2_similarity)

    query1_score = torch.sigmoid(query1_logit)
    query2_score = torch.sigmoid(query2_logit)
    print(query1_score.shape)

    query1_score = query1_score.squeeze(-1).view(B, seg, T)
    query2_score = query2_score.squeeze(-1).view(B, seg, T)

    return query1_score, query2_score


if __name__ == "__main__":
    
    # Dummy Data
    seg_len = torch.tensor([171, 127, 186, 129,  73, 159, 158, 194, 136,  28, 163, 176, 135,  49,
        184, 199, 199,  21, 101, 195], dtype=torch.int32)
    visual_feature = torch.randn((1,20,200, 2048), dtype=torch.float32) # B, seg, T, C
    q1 = torch.randn((1,300), dtype=torch.float32) # B, C
    q2 = torch.randn((1,300), dtype=torch.float32) # B, C

    config = load_json("./config/config.json")
    model = CHAN(config)
    q1_score, q2_score = model(visual_feature, seg_len.unsqueeze(0), q1, q2)
    print(q1_score.shape)


