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

LR = 0.001

ITER = 4000000
BUFFER_LIMIT = 1000
BATCH_SIZE = 2

height = int(args.height)
width = int(args.weight)
agent_num = int(args.agent_num)

if args.cuda != 'default':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    cuda = True

buff = ReplayBuffer(agent_num, height, width, 128)

model = DQN(height, width)
criterion = nn.MSELoss()
optim = RMSprop(model.parameters(), lr=LR)

if args.keep_train != 'default':
    model.load_state_dict(torch.load(sys.argv[1]))

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

    states = states.view(agent_num, 4, BATCH_SIZE, 3, 32, 32)

    action = action.long().view(BATCH_SIZE, agent_num, 1)
    scores = model(states, action, agent_num)

    scores = torch.cat(scores).view(BATCH_SIZE, agent_num, -1)

    scores = scores.gather(2, action)
    scores = torch.mean(scores, 1)

    loss = criterion(scores, reward)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if (i + 1) % 100 == 0:
        print('Iter:%d | loss:%.4f' %(i + 1, loss.item()))

    if (i + 1) % 1000 == 0:
        torch.save(model.state_dict(), sys.argv[1])

