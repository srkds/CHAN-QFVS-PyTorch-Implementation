import torch


class CHANModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1d_1= torch.nn.Conv1d(2048,512,kernel_size=5,stride=1,padding=2)
        self.max_pool_1 = torch.nn.MaxPool1d(2, stride=2, padding=0)

        self.conv1d_2 = torch.nn.Conv1d(512, 256, kernel_size=5,stride=1,padding=2)
        self.max_pool_2 = torch.nn.MaxPool1d(2, stride=2, padding=0)

        """
        for kernel size = 4 it was giving 2780 shots as original length but actually its 2783
        Error shape '[10, 2780, 1]' is invalid for input of size 27830
        so changing kernel_size 4 to 5
        """
        self.deconv1d_1 = torch.nn.ConvTranspose1d(256, 1024,kernel_size=5,stride=2,padding=1)
        self.deconv1d_2 = torch.nn.ConvTranspose1d(1024, 1024,kernel_size=5,stride=2,padding=1)
        
        # self.linear1 = torch.nn.Linear(512, 1000, bias=False)
        
        self.linear1 = torch.nn.Linear(1024, 1000, bias=False)
        self.linear2 = torch.nn.Linear(300, 1000, bias=False)
        self.MLP = torch.nn.Linear(1000, 1) 

    def forward(self, features, query1, query2):
        # Batch*segment or no_of features X no_of_feature_dim X no_of_shots
        """
        # without segments
        # original dimention is 1 X no_of_shots X 2048
        # for input to conv1d transposing the no_of_shots and feature dimentions that will result in each feature vector become feature column
        # transposed shape 1 X 2048 X no_of_shots
        """
        # print("ip feature shape, ", features.shape) [10, 2783, 2048]
        c1 = self.conv1d_1(features.view(1*features.shape[0],features.shape[1],-1).transpose(1,2))
        c1 = self.max_pool_1(c1) 

        c2 = self.conv1d_2(c1)
        c2 = self.max_pool_2(c2)

        d1 = self.deconv1d_1(c2)
        d2 = self.deconv1d_2(d1)
        # similarity1 = self.linear1(c1.transpose(1,2).view(features.shape[0], features.shape[1], -1))
        # print("d2 ", d2.shape)
        # print("s1 input shape: ", d2.transpose(1,2).shape)
        similarity1 = self.linear1(d2.transpose(1,2))
        
        query_1 = self.linear2(query1)
        query_2 = self.linear2(query2)

        joint_query1 = query_1.unsqueeze(1)*similarity1 # unsqueeze because concept is just in (1,1000) * (20,1000) wont happen thats why
        joint_query2 = query_2.unsqueeze(1)*similarity1

        q1_score = self.MLP(joint_query1)
        q2_score = self.MLP(joint_query2)
        
        q1_score = torch.sigmoid(q1_score)
        q2_score = torch.sigmoid(q2_score)

        return q1_score, q2_score

