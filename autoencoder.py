from model import *
from utils import get_args
import RPbuffer_v4
from RPbuffer_v4 import ReplayBuffer

import os
import sys

import numpy as np
import torch
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

args = get_args()

keep_train = True
cuda = False

LR = 0.001

ITER = 4000000
BATCH_SIZE = 8
TARGET_UPDATE = 10

height = int(args.height)
width = int(args.width)
agent_num = int(args.agent_num)

if args.cuda != 'default':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    cuda = True

buff = ReplayBuffer(agent_num, height, width, AE(), args.model_path, 10000, cuda)

model = AE()
embed = Embedding()
criterion = nn.MSELoss()
optim = Adam(model.parameters(), lr=LR)
embed_optim = Adam(embed.parameters(), lr=LR)

if args.keep_train != 'default':
    print('keep training, modelpath: {0}'.format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))
    embed.load_state_dict(torch.load('model/embed.pkl'))

if cuda:
    model.cuda()
    embed.cuda()
    criterion.cuda()

action = torch.zeros([BATCH_SIZE, agent_num])

for i in range(ITER):
    buff.collect(model, 0)

    if len(buff.memory) < BATCH_SIZE: continue

    batch = [random.choice(buff.memory) for j in range(BATCH_SIZE)]

    

    states = torch.cat([batch[j].states.unsqueeze(0) for j in range(BATCH_SIZE)])
    next_states = torch.cat([batch[j].next_state.unsqueeze(0) for j in range(BATCH_SIZE)])
    action = torch.cat([batch[j].action.unsqueeze(0) for j in range(BATCH_SIZE)])

    if cuda:
        states = states.cuda()
        action = action.cuda()
        next_states = next_states.cuda()

    action = action.long().view(BATCH_SIZE, agent_num, 1)

    state_vec = model(states)
    next_state_vec = model(next_states)

    action_vec = embed(action)
    state_vec  = state_vec + action_vec

    

    loss = criterion(state_vec, next_state_vec)
    optim.zero_grad()
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optim.step()
    embed_optim.step()

    if (i + 1) % 100 == 0:
        print('Iter:%d | loss:%.4f' %(i + 1, loss.item()))
    
    if (i + 1) % 100 == 0:
        torch.save(model.state_dict(), args.model_path)
        torch.save(embed.state_dict(), 'model/embed.pkl')
