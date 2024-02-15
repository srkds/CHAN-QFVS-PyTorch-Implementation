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

    def __init__(self):
        self._build_dataset()
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self.p = 0
        self.r = 0
        self.f1 = 0

    def _build_model(self):
        self.model = CHANModel()

    def _build_dataset(self):
        self.dataset = UTEDataset()

    def _build_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=10, shuffle=True)

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def output(self):
        print(f"p: {self.p} | r: {self.r} | f1: {self.f1}")

    def train(self):
        loss_list = []
        for epoch in range(5):
            self.model = self.model.train()
            # loss = torch.zeros(1)
            for batch_idx, (features, q1, q2, q1gt, q2gt, q1t, q2t) in enumerate(self.dataloader):
                # print("Batch_id: ", batch_idx+1)
                # print(features.shape)
                # print(q1t)
                # print(q2t)
                q1_pred, q2_pred = self.model(features, q1, q2)
                # print("q1_preds: ", q1_pred)
                # print("q1_gt: ", q1gt)
                # print("q2_preds: ", q2_pred)
                # print("q2_gt: ", q2gt)
                # print("q1_gt: ", q1_pred.shape) # [10, 2783, 1]
                # print("q1_gt: ", q1gt.shape) # [10, 2783]
                
                q1_loss = F.binary_cross_entropy(q1_pred, q1gt.view(q1_pred.shape))
                q2_loss = F.binary_cross_entropy(q2_pred, q2gt.view(q2_pred.shape))
                loss = q1_loss+q2_loss
                print("Loss: ", loss)
                loss_list.append(loss.data)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.evaluate(1, 0.02)

    def evaluate(self, test_video, top_percent):
        """
        method takes test_video id and top_percent which is criteria for selecting a shot like 0.28 etc.
        """
        f1_sum = 0
        p_sum = 0
        r_sum = 0

        # loading pickle file that is pretrained model to get features of query(text)
        embedding = load_pickle("./data/query_dictionary.pkl") 

        evaluation_num = 0

        # directory ../data/P01 contains Oracle summaries in txt file format
        # each concept1_concept2_oracle.txt file contains GT summary shot numbers
        for _, _, files in os.walk("./data/P01"): # getting all files in given folder, each file meaning single training example query.
            evaluation_num = len(files)
            # count = 0
            for file in files: # looping through each file or we can say training example that is concept1_concept2_oracle.txt 

                # to only evalueate on 6 queries as test set instead of whole train set thats 45 
                # count += 1
                # if count > 6:
                #     break
                # GTSummary list generation converting numbers given in file to list like [2, 3, ... ,443]
                summaries_GT = []
                with open("./data/P01/"+file, "r") as f:
                    for line in f.readlines():
                        # print("line: ",line)
                        # print("line strip:",int(line.strip()))
                        summaries_GT.append(int(line.strip())) # this will append each GT shot number converting it to int and making its list like [1, 3,..434]
                    # print(summaries_GT)

                f = h5py.File('./data/features/V1_resnet_avg.h5', 'r') # loading video features
                features = torch.tensor(f["feature"][()], dtype=torch.float32) # converting feature values to tensor and loading features

                transfer={"Cupglass":"Glass","Musicalinstrument":"Instrument","Petsanimal":"Animal"}

                query1, query2 = file.split('_')[:2]
                if query1 in transfer:
                    query1=transfer[query1]
                if query2 in transfer:
                    query2=transfer[query2]

                query1 = torch.tensor(embedding[query1], dtype=torch.float32)
                query2 = torch.tensor(embedding[query2], dtype=torch.float32)
                # print(len(features))
                mask = torch.zeros(1, 1, len(features), dtype=torch.bool)
                for i in range(1):
                    for j in range(1):
                        for k in range(len(features)):
                            mask[i][j][k]=1
                q1_score, q2_score = self.model(features.unsqueeze(0), query1.unsqueeze(0), query2.unsqueeze(0))
                # print("q1:", q1_score.shape)
                # q1_score dims [1, 2783, 1]
                
                score = q1_score + q2_score 
                print(score.shape)  # dims [1, 2783, 1]
                score = score.transpose(1,2)
                score = score.masked_select(mask) # dims [7745089]
                
                _,top_index=score.topk(int(score.shape[0]*0.02))
                    
                top_index +=1
                print(file.split('_')[:2])
                print(top_index)
                video_shots_tag = load_videos_tag(mat_path="./data/evaluation/Tags.mat")
                p, r, f1 = calculate_semantic_matching(list(top_index.cpu().numpy()), summaries_GT, video_shots_tag, video_id=1)

                f1_sum += f1
                p_sum += p
                r_sum += r
                print(p, r , f1)
                # self.dataset.append(file[:file.find("_oracle.txt")]+"_"+"1")
            print("p ", p_sum/evaluation_num, "r ", r_sum/evaluation_num, "f1 ", f1_sum/evaluation_num)
            # print("p ", p_sum/6, "r ", r_sum/6, "f1 ", f1_sum/6)
        