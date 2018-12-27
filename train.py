from model import *
from utils import get_args
from RPbuffer_v3 import ReplayBuffer

import os
import sys

import numpy as np
import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader
import torch.nn.functional as F

args = get_args()

keep_train = True
cuda = False

LR = 0.001

ITER = 4000000
BATCH_SIZE = 1
TARGET_UPDATE = 10

height = int(args.height)
width = int(args.width)
agent_num = int(args.agent_num)

if args.cuda != 'default':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    cuda = True

buff = ReplayBuffer(agent_num, height, width, DQN(), args.model_path, 10000, cuda)

model = DQN()
target_model = DQN()
criterion = nn.MSELoss()
optim = RMSprop(model.parameters(), lr=LR)

if args.keep_train != 'default':
    print('keep training, modelpath: {0}'.format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))

if cuda:
    model.cuda()
    target_model.cuda()
    criterion.cuda()

action = torch.zeros([BATCH_SIZE, agent_num])

for i in range(ITER):
    buff.collect(model, 0)

    if len(buff.memory) < BATCH_SIZE: continue

    batch = [random.choice(buff.memory) for j in range(BATCH_SIZE)]

    

    states = torch.cat([batch[j].states.unsqueeze(0) for j in range(BATCH_SIZE)])
    action = torch.cat([batch[j].action.unsqueeze(0) for j in range(BATCH_SIZE)])
    reward = torch.cat([batch[j].reward.unsqueeze(0) for j in range(BATCH_SIZE)]).unsqueeze(2)
    done = [batch[j].done for j in range(BATCH_SIZE)]

    if cuda:
        states = states.cuda()
        action = action.cuda()
        reward = reward.cuda()

    action = action.long().view(BATCH_SIZE, agent_num, 1)

    pred_scores = model(states[:,:3]).gather(2, action) #(batch, agent, 1)

    if cuda:
        target_scores = torch.zeros([BATCH_SIZE, agent_num, 5], dtype=torch.float32).cuda()
    else:
        target_scores = torch.zeros([BATCH_SIZE, agent_num, 5], dtype=torch.float32)

    for j in range(BATCH_SIZE):
        if done[j] is not True:
            target_scores[j] = target_model(states[j,1:].unsqueeze(0)).squeeze(0).max(1)[0].view(-1) #(batch, agent, 1)

    # print(reward)

    target_scores = target_scores * 0.999 + reward

    loss = F.smooth_l1_loss(pred_scores, target_scores)
    optim.zero_grad()
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optim.step()

    

    if i % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())

    if (i + 1) % 100 == 0:
        print('Iter:%d | loss:%.4f | pred_scores:%.4f | target_scores:%.4f' %(i + 1, loss.item(), torch.mean(pred_scores[0]).item(), torch.mean(target_scores[0]).item()))
    
    if (i + 1) % 100 == 0:
        torch.save(model.state_dict(), args.model_path)
