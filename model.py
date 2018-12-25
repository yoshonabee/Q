import os
import glob
import random

from utils import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)

        self.linear = nn.Linear(3072, 5)

    def forward(self, states):
        #shape of states: (batch, state_num, agent, 3, height, width)
        scores = []
        for agent in range(states.shape[2]):
            s1 = states[:,0,agent,:] #(batch, 3, 32, 32)
            s2 = states[:,1,agent,:] #(batch, 3, 32, 32)
            s3 = states[:,2,agent,:] #(batch, 3, 32, 32)

            x = torch.cat([self.conv(s1), self.conv(s2), self.conv(s3)], 1)
            x = self.linear(x).unsqueeze(1) #(batch, 1, 5)
            scores.append(x)

        scores = torch.cat(scores, 1) #(batch, agent, 5)
        return scores

    def conv(self, s):
        x = F.relu(self.maxp1(self.conv1(s)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))

        return x.view(x.size(0), -1)
