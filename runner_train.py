import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import h5py

from model import CHANModel
from dataset import UTEDataset
from sementic_evaluate import load_videos_tag, calculate_semantic_matching
from utils import load_pickle


class Runner():
  def __init__(self, config, train_videos, test_video):
    self.config = config
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.train_videos = train_videos
    self.test_video = test_video
    self._build_model()
    self._build_dataloader()
    self._build_optimizer()
    self.max_p = 0
    self.max_r = 0
    self.max_f1 = 0

  def _build_model(self):
    self.model = CHAN(self.config)

  def _build_dataset(self):
    return UTEDataset(self.train_videos)

  def _build_dataloader(self):
    dataset = self._build_dataset()
    self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])

  def _build_optimizer(self):
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["decay_rate"])

  def output(self):
    print(f"max_p = {self.max_p}, max_r = {self.max_r}, max_f1 = {self.max_f1}")

  def train(self):

    self.model.train()

    print("start to evaluate random result")
    self.evaluate(self.test_video, self.config["top_percent"])
    print("end to evaluate random result")
    for epoch in range(self.config["epoch"]):

      for video_feature, seg_len, query1_emd, query2_emd, query1_gt, query2_gt, mask_gt, query1_t, query2_t in self.dataloader:

        train_num = seg_len.shape[0]
        self.optimizer.zero_grad()

        mask = torch.zeros(train_num, self.config["max_segment_num"], self.config["max_frame_num"], dtype=torch.bool)
        for i in range(train_num):
          for j in range(len(seg_len[i])):
            for k in range(seg_len[i][j]):
              mask[i][j][k] = 1

        # (B, seg, T) -> (5, 20, 200)
        query1_score, query2_score = self.model(video_feature, seg_len, query1_emd, query2_emd)

        loss = torch.zeros(1)
        for i in range(train_num):
          query1_score_tmp = query1_score[i].masked_select(mask[i]).unsqueeze(0) # (B, seg, T) (5, 20, 200) -> (1, T) -> (1, 2152)
          query2_score_tmp = query2_score[i].masked_select(mask[i]).unsqueeze(0) # same

          query1_gt_tmp = query1_gt[i].masked_select(mask_gt[i]).unsqueeze(0)
          query2_gt_tmp = query2_gt[i].masked_select(mask_gt[i]).unsqueeze(0)

          loss1 = F.binary_cross_entropy(query1_score_tmp, query1_gt_tmp)
          loss2 = F.binary_cross_entropy(query2_score_tmp, query2_gt_tmp)

          loss += loss1+loss2

        loss.backward()
        self.optimizer.step()
      self.evaluate(self.test_video, self.config["top_percent"])


  def evaluate(self, video_id, top_percent):
    self.model.eval()

    f1_sum, p_sum , r_sum = 0,0,0

    embedding = load_pickle("./data/query_dictionary.pkl")

    evaluation_num = 0
    for _,_,files in os.walk(f"./data/Oracle_summaries/P0{video_id}"):
      evaluation_num = len(files)
      for file in files:
        summaries_GT=[]
        with open(f"./data/Oracle_summaries/P0{video_id}/{file}","r") as f:
          for line in f.readlines():
            summaries_GT.append(int(line.strip()))

        f = h5py.File(f'./data/processed/V{video_id}_resnet_avg.h5', 'r')
        features=torch.tensor(f["features"][()], dtype=torch.float32)
        seg_len=torch.tensor(f["seg_len"][()], dtype=torch.int32)
        # features = torch.tensor(f["feature"][()], dtype=torch.float32) # converting feature values to tensor and loading features

        transfer={"Cupglass":"Glass","Musicalinstrument":"Instrument","Petsanimal":"Animal"}

        query1, query2 = file.split('_')[:2]
        if query1 in transfer:
            query1=transfer[query1]
        if query2 in transfer:
            query2=transfer[query2]

        query1 = torch.tensor(embedding[query1], dtype=torch.float32)
        query2 = torch.tensor(embedding[query2], dtype=torch.float32)
        # print(len(features))
        mask = torch.zeros(1, 20, 200, dtype=torch.bool)
        for i in range(1):
            for j in range(len(seg_len.unsqueeze(0)[i])):
                for k in range(seg_len.unsqueeze(0)[i][j]):
                    mask[i][j][k]=1

        q1_score, q2_score = self.model(features.unsqueeze(0), seg_len.unsqueeze(0), query1.unsqueeze(0), query2.unsqueeze(0))
        score = q1_score + q2_score
        score = score.masked_select(mask)

        _, top_index = score.topk(int(score.shape[0]*top_percent))

        if 3588 in top_index and video_id == 4:
            print("removing 3588")
            mk = top_index != 3588
            top_index = top_index[mk]

        
        # print("before summaries_GT.shape : ", summaries_GT)
        if 3692 in summaries_GT and video_id == 2:
            # removing this index from video 2 using remove() method because summaries_GT is python list and not the tensor
            print("removing 3692")
            summaries_GT.remove(3692)

        top_index_to_print = top_index + 1 
        # print(file.split('_')[:])
        # print("evaluation video: ", video_id-1)
        # print(top_index)
        # print(top_index_to_print)
        video_shots_tag = load_videos_tag(mat_path="./data/evaluation/Tags.mat")
        p, r, f1 = calculate_semantic_matching(list(top_index.cpu().numpy()), summaries_GT, video_shots_tag, video_id=video_id-1) # video id = 0,1,2,3

        f1_sum += f1
        p_sum += p
        r_sum += r
        print(p, r , f1)
        # self.dataset.append(file[:file.find("_oracle.txt")]+"_"+"1")
    print("p ", p_sum/evaluation_num, "r ", r_sum/evaluation_num, "f1 ", f1_sum/evaluation_num)

    self.model.train()
