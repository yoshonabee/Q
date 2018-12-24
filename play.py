import os
import sys
import datetime
import numpy as np
import torch
from utils import *
from model import *
from RPbuffer_v3 import *

model = DQN(32, 32)
# model.load_state_dict(torch.load('model/q.pkl', map_location='cpu'))

buff = ReplayBuffer(3, 32, 32, 128)

buff.play(model, 128)