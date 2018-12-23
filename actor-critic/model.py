import os
import glob
import random

from utils import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class ActorCritic(nn.Module):
    def __init__(self, height, width):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=2) # 1*n*n -> 32*(n+2)*(n+2)
        self.maxp1 = nn.MaxPool2d(2, 2) # -> 32*(n/2 + 1)*(n/2 + 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1) # -> 32*(n/2 + 1)*(n/2 + 1)
        self.maxp2 = nn.MaxPool2d(2, 2) # -> 32*(n/4)*(n/4)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1) # -> 64*(n/4)*(n/4)
        self.maxp3 = nn.MaxPool2d(2, 2) # -> 64*(n/8)*(n/8)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1) # -> 64*(n/8)*(n/8)
        self.maxp4 = nn.MaxPool2d(2, 2) # -> 64*(n/16)*(n/16)
        # 64 * 16 * 16
        self.avgp = nn.AvgPool2d(height//16//4, width//16//4)
        # self.avgp = nn.AveragePool2d(height//16, width//16, stride=1, padding=1)

        self.lstm = nn.LSTMCell(1024, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 5)

    def forward(self, inputs, hx, cx):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = self.avgp(x)
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        return self.critic_linear(hx).view(-1), self.actor_linear(hx), (hx, cx)

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
        last_state = data[0].astype(np.float32)
        last_hx = data[1]
        last_cx = data[2]
        cmd = data[3]
        last_score = np.array([data[4] for i in range(data[0].shape[0])]).reshape(-1).astype(np.float32)
        state = data[5].astype(np.float32)
        hx = data[6]
        cx = data[7]
        score = np.array([data[8] for i in range(data[0].shape[0])]).reshape(-1).astype(np.float32)
        return last_state, last_hx, last_cx, cmd, last_score, state, hx, cx, score

class ActorCriticLoss(nn.Module):
    def __init__(self, beta = 0.001):
        super(ActorCriticLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.BCELoss = nn.BCELoss(reduction='none')
        self.beta = beta

    def forward(self, pred, target, score):
        cmd_pred, r_pred = pred[0], pred[1]
        cmd, r = target[0], target[1]

        cmd = torch.max(cmd, 1)[1]

        actor_loss = -r * self.CrossEntropyLoss(cmd_pred, cmd).view(r.shape)
        # print(actor_loss.shape)

        critic_loss = self.L1Loss(r, r_pred)
        # print(critic_loss.shape)

        reg = self.MSELoss(torch.softmax(cmd_pred, 0), torch.full_like((cmd_pred), 0.2, dtype = torch.float32))
        # print(reg.shape)
        # print(torch.mean(actor_loss).item(), torch.mean(reg).item())
        # print(torch.mean(actor_loss).item(), torch.mean(critic_loss).item())
        loss = torch.mean(actor_loss + critic_loss + self.beta * reg)
        return loss

    def L1Loss(self, y, target):
        loss = (y - target)
        loss = torch.abs(loss)
        return loss

    def MSELoss(self, y, target):
        loss = (y - target) ** 2
        loss = torch.mean(loss, 1)
        loss = torch.sqrt(loss)
        return loss


