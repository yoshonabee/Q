from model import *

import os
import sys

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

if len(sys.argv) < 3:
    print("usage: python3 train.py [model weight] [height] [width] [cuda]")
    exit(0)

keep_train = True
cuda = False
buffer_path = './buffer'

LR = 0.01
M = 0.9
# BATCH_SIZE = int(sys.argv[2])
ITER = 4000000
BUFFER_LIMIT = 1000
IOLIMIT = 50

height = int(sys.argv[2])
width = int(sys.argv[3])

if sys.argv[2] != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]
    cuda = True

buff = Buffer(buffer_path, ITER, BUFFER_LIMIT, IOLIMIT)
buffLoader = DataLoader(buff, batch_size=1)


model = ActorCritic(height, width)
model_pred = ActorCritic(height, width)

criterion = ActorCriticLoss()
optim = SGD(model.parameters(), lr=LR, momentum = M)

if keep_train:
    model.load_state_dict(torch.load(sys.argv[1]))

if cuda:
    model.cuda()
    model_pred.cuda()
    criterion.cuda()

for i, (last_state, last_hx, last_cx, cmd, last_score, state, hx, cx, score) in enumerate(buffLoader):
    if cuda:
        last_state = last_state.cuda()
        last_hx = last_hx.cuda()
        last_cx = last_cx.cuda()
        cmd = cmd.cuda()
        last_score = last_score.cuda()
        state = state.cuda()
        hx = hx.cuda()
        cx = cx.cuda()
        score = score.cuda()

    last_state = torch.squeeze(last_state, 0)
    last_hx = torch.squeeze(last_hx, 0)
    last_cx = torch.squeeze(last_cx, 0)
    cmd = torch.squeeze(cmd, 0)
    last_score = torch.squeeze(last_score, 0)
    state = torch.squeeze(state, 0)
    hx = torch.squeeze(hx, 0)
    cx = torch.squeeze(cx, 0)
    score = torch.squeeze(score, 0)

    optim.zero_grad()
    model_pred.load_state_dict(model.state_dict())

    last_score_pred, cmd_pred, _ = model(last_state, last_hx, last_cx)
    score_pred, _, _ = model_pred(state, hx, cx)

    r = score - last_score
    r_pred = score_pred - last_score_pred

    loss = criterion((cmd_pred, r_pred), (cmd, r), score)
    loss.backward()

    if (i + 1) % 100 == 0:
        print('Iter:%d | loss:%.4f' %(i + 1, loss.item()))

    if (i + 1) % 1000 == 0:
        torch.save(model.state_dict(), sys.argv[1])

