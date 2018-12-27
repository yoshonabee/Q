import os
import sys
import datetime
import numpy as np
import torch
from utils import *
from model import *
from RPbuffer_v3 import *

cuda = False

args = get_args()

if args.cuda != 'default':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    cuda = True

model = DQN(cuda)
agent_num = int(args.agent_num)
height = int(args.height)
width = int(args.width)

model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

buff = ReplayBuffer(agent_num, height, width, DQN(cuda), 'model/qq.pkl', 128, False)

buff.play(model, 128)
# buff.save_state('./')