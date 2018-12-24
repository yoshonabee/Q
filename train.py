from model import *
from utils import get_args
from RPbuffer_v3 import ReplayBuffer

import os
import sys

import numpy as np
import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader

args = get_args()

keep_train = True
cuda = False

LR = 0.0001

ITER = 4000000
BUFFER_LIMIT = 1000
BATCH_SIZE = 1
TARGET_UPDATE = 10

height = int(args.height)
width = int(args.weight)
agent_num = int(args.agent_num)

if args.cuda != 'default':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    cuda = True

buff = ReplayBuffer(agent_num, height, width, DQN(), args.model_path, 128)

model = DQN()
target_model = DQN()
criterion = nn.MSELoss()
optim = RMSprop(model.parameters(), lr=LR)

if args.keep_train != 'default':
    print(f'keep training, modelpath: {args.model_path}')
    model.load_state_dict(torch.load(args.model_path))

if cuda:
    model.cuda()
    criterion.cuda()

action = torch.zeros([BATCH_SIZE, agent_num])

for i in range(ITER):
    buff.collect()
    batch = [random.choice(buff.memory) for j in range(BATCH_SIZE)]

    states = torch.cat([batch[j].states.unsqueeze(0) for j in range(BATCH_SIZE)])
    action = torch.cat([batch[j].action.unsqueeze(0) for j in range(BATCH_SIZE)])
    reward = torch.tensor([batch[j].reward for j in range(BATCH_SIZE)]).view(-1, 1)
    
    if cuda:
        states = states.cuda()
        action = action.cuda()
        reward = reward.cuda()

    action = action.long().view(BATCH_SIZE, agent_num, 1)
    pred_scores = model(states[:,0:3]).gather(2, action) #(batch, agent, 1)
    pred_scores = torch.mean(pred_scores.view(-1, 1), 1) #(batch, 1)
    target_scores = target_model(states[:,1:4]).max(2)[0].view(-1, 1) #(batch, agent, 1)
    target_scores = torch.mean(target_scores.view(-1, 1), 1) #(batch, 1)

    pred_scores = pred_scores * 0.999 + reward

    loss = criterion(pred_scores, target_scores)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())

    if (i + 1) % 100 == 0:
        print('Iter:%d | loss:%.4f' %(i + 1, loss.item()))

    if (i + 1) % 1000 == 0:
        torch.save(model.state_dict(), args.model_path)

