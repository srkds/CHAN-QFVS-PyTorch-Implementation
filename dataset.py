import torch
from torch.utils.data import Dataset
import h5py
import os

from utils import load_pickle


# Custom class to load the dataset
class UTEDataset(Dataset):
    def __init__(self):
        """
        Constructor: will be called when UTEDataset initialized
        Here we are creating a training example per query.
        It will load all the queries, and embedding.
        """
        self.dataset = []
        for _, _, files in os.walk("./data/P01"):
            for file in files:
                self.dataset.append(file[:file.find("_oracle.txt")]+"_"+"1")
        self.embedding=load_pickle("./data/query_dictionary.pkl")
                
    def __getitem__(self, index):
        """
        This method will be callded when access indexed.
        """
        video_id = self.dataset[index].split("_")[2] 
        f = h5py.File('./data/features/V1_resnet_avg.h5', 'r') # loading features
        features=torch.tensor(f["feature"][()], dtype=torch.float32)
        # Commenting below code for experimenting on whole video
        # features=features[:20] # working only on the small video clips which is 20 shots 🟥

        query_1, query_2 = self.dataset[index].split("_")[0:2]
        query_1_GT = torch.zeros(features.shape[0]) # extending GT from 20 to full video shot length
        query_2_GT = torch.zeros(features.shape[0])

        transfer={"Cupglass":"Glass",
                  "Musicalinstrument":"Instrument",
                  "Petsanimal":"Animal"}

        with open("./data/Dense_per_shot_tags/P01/P01.txt", "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                queries = line.strip().split(',')
                if query_1 in queries:
                    query_1_GT[index] = 1
                if query_2 in queries:
                    query_2_GT[index] = 1
                # NO need of the condition given below 🟥
                # if index == 19:
                #     break

        if query_1 in transfer:
            query_1=transfer[query_1]
        if query_2 in transfer:
            query_2=transfer[query_2]

        q1_text = query_1
        q2_text = query_2
        # print("Q1: ", query_1)
        # print("Q2: ", query_2)
        query_1 = torch.tensor(self.embedding[query_1], dtype=torch.float32)
        query_2 = torch.tensor(self.embedding[query_2], dtype=torch.float32)
        
        return features, query_1, query_2, query_1_GT, query_2_GT, q1_text, q2_text

    def __len__(self):
        return len(self.dataset)

if __name__=="__main__":
    dataset = UTEDataset()
    print(f"size of dataset is {len(dataset)}")