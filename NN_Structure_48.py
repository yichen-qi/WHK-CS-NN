import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, af_output_flag = 1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        # self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 48)
        self.af_output_flag = af_output_flag

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.af_output_flag == 1:
            x = F.softplus(self.fc4(x))
        else:   
            x = self.fc4(x)
        return x
    

class MSEWithHingePenalty(nn.Module):
    def __init__(self, penalty_weight=0.1):  
        super(MSEWithHingePenalty, self).__init__()
        self.mse = nn.MSELoss()
        self.penalty_weight = penalty_weight

    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)

        hinge_penalty = torch.clamp(-outputs, min=0).sum()

        total_loss = mse_loss + self.penalty_weight * hinge_penalty
        return total_loss