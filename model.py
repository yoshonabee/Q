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
    def __init__(self, height, width):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)

        self.linear = nn.Linear(2048, 5)

    def forward(self, states, action, agent_numbers):
        scores = []

        for i in range(agent_numbers):
            x = torch.cat([self.conv(states[i][:][s]) for s in range(4)], 1)
            x = self.linear(x).unsqueeze(0)
            scores.append(x)

        return scores

    def conv(self, s):
        x = F.relu(self.maxp1(self.conv1(s)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))

        return x.view(x.size(0), -1)

class Buffer(Dataset):
    def __init__(self, path = './buffer', max_iter=100000, buffer_limit=1000, IOlimit=100):
        self.path = path
        self.max_iter = max_iter
        self.buffer_limit = buffer_limit
        self.IOlimit = IOlimit

    def __getitem__(self, index):
        data_list = glob.glob('{0}/*'.format(self.path))

        if (len(data_list) > self.buffer_limit):
            data_list.sort()
            for i in range(self.IOlimit):
                os.remove(data_list[i])
            data_list = data_list[self.IOlimit:]

        data_name = random.choice(data_list)
        data = load_object(data_name)
        return self.transform(data)

    def __len__(self):
        return self.max_iter

    def transform(self, data):
        s1 = data[0].astype(np.float32)
        s2 = data[1].astype(np.float32)
        s3 = data[2].astype(np.float32)
        s4 = data[3].astype(np.float32)
        action = data[4].astype(np.int64)

        return s1, s2, s3, s

def action_select():
    pass


