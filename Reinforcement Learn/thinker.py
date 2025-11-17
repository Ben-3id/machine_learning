import torch
import torch.nn as nn
from torch import optim
import os

class Qnet(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.networks=nn.Sequential( # input ---> 128 ---> 64 ---> output
            nn.Linear(input,128),
            nn.ReLU(),    
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,output)
        )
    
    def forward(self,X):
        return self.networks(X)
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrain:
    def __init__(self,model,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.model=model
        self.optimizer=torch.optim.Adam(params=model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self,state,action,reward,next_state,done):
        state=torch.tensor(state,dtype=torch.float)
        action=torch.tensor(action,dtype=torch.long)
        reward=torch.tensor(reward,dtype=torch.float)
        next_state=torch.tensor(next_state,dtype=torch.float)
        

        if len(state.shape)==1:
            state=torch.unsqueeze(state,0)
            action=torch.unsqueeze(action,0)
            reward=torch.unsqueeze(reward,0)
            next_state=torch.unsqueeze(next_state,0)
            done=(done, )

        pred=self.model(state)
        target=pred.clone()
        for idx in range(len(done)):
            Q_new=reward[idx] 
            if not done[idx]:
                Q_new=reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        self.optimizer.zero_grad()
        loss=self.criterion(pred,target.detach())
        loss.backward()
        self.optimizer.step()



