import torch
from torch.utils.data import Dataset
import h5py
import os

from utils import load_pickle


# Custome Dataset Class
class UTEDataset(Dataset):
  """
  A custome dataset for UTEgocentric

  Attributes
  ----------
  dataset : list
    a list of queries and accociated video eg. Lady_Food_1

  """

  def __init__(self, videos):
    """
    Creating dataset list for all training examples like
    0 q1_q2_video_id
    1 q1_q2_video_id
    2 q1_q2_video_id

    Parameters
    ----------
    videos : list
      list of video ids for training examples eg. [1, 2, 3]
    ...
    """
    self.dataset = []
    train_videos = videos
    for video_id in train_videos:
      for _, _, files in os.walk(f"./data/Oracle_summaries/P0{video_id}"):
        for file in files:
          self.dataset.append(file[:file.find("_oracle.txt")]+f"_{video_id}") # from q1_q2_oracle.txt -> q1_q2_vid_id

    # pretraind glove vectors embeddings for text
    self.embedding = load_pickle("./data/query_dictionary.pkl")

  def list_dataset(self):
    return self.dataset

  def __getitem__(self, index):

    # get video id from indexed query eg. "Lady_Food_1" indicating lady and food as query and video is 1
    video_id = self.dataset[index].split("_")[2]

    # then load featuers of that video
    f = h5py.File(f"./data/processed/V{video_id}_resnet_avg.h5","r")
    features = torch.tensor(f["features"][()], dtype=torch.float32)
    seg_len = torch.tensor(f["seg_len"][()], dtype=torch.int32)

    query1, query2 = self.dataset[index].split("_")[0:2]

    query1_GT = torch.zeros(20*200) # 20 segments each having 200 shots
    query2_GT = torch.zeros(20*200)

    transfer = {
        "Cupglass":"Glass",
        "Musicalinstrument":"Instrument",
        "Petsanimal":"Animal"
    }

    with open(f"./data/Dense_per_shot_tags/P0{video_id}/P0{video_id}.txt", "r") as f:
      lines = f.readlines()
      for index, line in enumerate(lines):
        queries = line.strip().split(",")
        if query1 in queries:
          query1_GT[index] = 1
        if query2 in queries:
          query2_GT[index] = 1


    shot_num = seg_len.sum() # get count of total no of shots
    mask_GT = torch.zeros(20*200, dtype=torch.bool)
    for i in range(shot_num):
      mask_GT[i] = 1

    if query1 in transfer:
      query1 = transfer[query1]
    if query2 in transfer:
      query2 = transfer[query2]

    query1_emd = torch.tensor(self.embedding[query1], dtype=torch.float32)
    query2_emd = torch.tensor(self.embedding[query2], dtype=torch.float32)

    return features, seg_len, query1_emd, query2_emd, query1_GT, query2_GT, mask_GT, query1, query2

  def __len__(self):
    return len(self.dataset)

if __name__=="__main__":
    test_dtst = UTEDataset([1,2,3])
    feat, seg_len, q1_emd, q2_emd, gt1, gt2, mgt, q1, q2 = test_dtst[10]
    print(f"size of dataset is {len(test_dtst)}")
    print(f"video feature shape: {feat.shape}")
    print(f"segment length     : {seg_len.shape}")
    print(f"text 1 emd shape   : {q1_emd.shape}")
    print(f"text 2 emd shape   : {q2_emd.shape}")
    print(f"gt1                : {gt1.shape}")
    print(f"gt2                : {gt2.shape}")
    print(f"mask gt            : {mgt.shape}")
    print(f"q1                 : {q1}")
    print(f"q2                 : {q2}")