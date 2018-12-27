import os
import sys
import datetime
import numpy as np
import torch
import random
from utils import *
from model import *
from lib.game.game import Game, Command
from lib.game.agent import Agent
from time import sleep
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------
#acting.py is the class for collect the training data from the game
#--------------------------------------------------------------------------------
RANDOM_THRES = 0.1


class Data:
    def __init__(self, states, action, reward, done):
        self.states = states
        self.action = action
        self.reward = reward
        self.done = done

class ReplayBuffer():
    def __init__(self, agent_num, height, width, model, modelpath, game_round, cuda=True):
        #initial the model in the replay buffer
        #model is be defined using pytorch lib

        #the game is the enviroment of route planning game in the project
        self.height = height
        self.width = width
        self.game_round = game_round
        self.agent_num = agent_num
        self.model = model
        self.modelpath = modelpath
        self.cuda = cuda

        self.states = torch.zeros([4, self.agent_num, 3, self.height, self.width], dtype=torch.float32)

        #default score setting in the game
        self.acquisition_sum = 400
        self.explored_target_sum = 70
        self.explored_sum = 40
        self.total_score = self.acquisition_sum + self.explored_target_sum + self.explored_sum
        self.time_decrease = -0.00005
        self.crash_time_penalty = -0.0001
        self.crash_sum = -400
        self.reg_val = 1

        # self.action = torch.tensor([random.randint(0, 4) for i in range(self.agent_num)])
        self.memory = []
        self.buffer_limit = 1000

        self.game = None
        self.score = None
        self.initialGame()

    def collect(self, model, verbose=1):
        if self.cuda:
            self.model.cuda()

        self.model.load_state_dict(model.state_dict())

        if self.game.active and self.game.state < self.game_round:
            #observe part of the bot
            action = self.select_action()
            
            score = np.array([self.game.tryOneRound(intoCommand(i, action[i])) for i in range(self.agent_num)])
            self.game.runOneRound([intoCommand(i, action[i]) for i in range(self.agent_num)])

            reward = score - self.score

            self.states[0] = self.states[1]
            self.states[1] = self.states[2]
            self.states[2] = self.states[3]
            s = np.array([self.game.outputAgentImage(i) for i in range(self.agent_num)]).astype(np.float32)
            s = torch.from_numpy(s)
            self.states[3] = s

            if verbose == 1:
                print(self.game.outputScore())

            done = False
            if self.game.active is False:
                done = True

            data = Data(self.states, action, torch.from_numpy(reward).float(), done)
            self.memory.append(data)

            self.score = np.array([self.game.outputScore()] * self.agent_num)

            if len(self.memory) > self.buffer_limit:
                self.memory.remove(self.memory[0])
        else:
            #only occur when the game is not active
            self.initialGame()

    def select_action(self):
        if random.random() < RANDOM_THRES + (100 / self.game.state) ** 1 / 2:
            return torch.tensor([random.randint(0, 4) for i in range(self.agent_num)])
        else:
            with torch.no_grad():
                if self.cuda:
                    return self.model(self.states[1:].cuda().unsqueeze(0)).squeeze(0).max(1)[1].view(self.agent_num).cpu()
                else:
                    return self.model(self.states[1:].unsqueeze(0)).squeeze(0).max(1)[1].view(self.agent_num)


    def save_state(self, path):
        for i in range(self.agent_num):
            img = np.zeros([32, 32, 3], dtype=np.uint8)
            arr = self.game.outputAgentImage(i)
            img[:,:,0] = arr[0,:,:]
            img[:,:,1] = arr[1,:,:]
            img[:,:,2] = arr[2,:,:]
            plt.subplot(2,2,i + 1)
            plt.imshow(img)
        plt.show()

    def play(self, model, game_round):
        while self.game.active and self.game.state < game_round:
            self.states[0] = self.states[1]
            self.states[1] = self.states[2]
            self.states[2] = self.states[3]
            s = np.array([self.game.outputAgentImage(i) for i in range(self.agent_num)]).astype(np.float32)
            s = torch.from_numpy(s)
            self.states[3] = s

            action = model(self.states[1:].unsqueeze(0)).max(2)[1].view(self.agent_num)
            print(action)
            action = [intoCommand(i, action[i]) for i in range(self.agent_num)]
            self.game.runOneRound(action)
            print(self.game.outputScore())
            sleep(0.1)

    def initialGame(self):
        #initial the game envirnment for th replay buffer
        t = (self.height + self.width) / 2

        self.game = Game(self.height, self.width, self.game_round)
        self.game.setRandomMap(self.agent_num, int(t * 0.3) ** 2, int(t * 0.1) ** 2)
        self.game.setScore(self.acquisition_sum, self.explored_target_sum, self.explored_sum, self.time_decrease, self.crash_time_penalty, self.crash_sum, self.reg_val)

        self.game.runOneRound([Command(i, 0, 0) for i in range(self.agent_num)])
        self.score = np.array([self.game.outputScore() for i in range(self.agent_num)])

        self.states = torch.zeros([4, self.agent_num, 3, self.height, self.width], dtype=torch.float32)
        
        s = np.array([self.game.outputAgentImage(i) for i in range(self.agent_num)]).astype(np.float32)
        s = torch.from_numpy(s)
        self.states[3] = s

        print('New Game!')

    def loadWeight(self):
        if os.path.exists(self.modelpath):
            if self.cuda:
                self.model.load_state_dict(torch.load(self.modelpath))
            else:
                self.model.load_state_dict(torch.load(self.modelpath, map_location='cpu'))
        else:
            print('Model weight [{0}] not found'.format(self.modelpath))
        return


def intoCommand(i, command):
    if command == 0: return Command(i, 0, 1)
    elif command == 1: return Command(i, 1, 0)
    elif command == 2: return Command(i, 0, -1)
    elif command == 3: return Command(i, -1, 0)
    return Command(i, 0, 0)