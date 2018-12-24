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
    def __init__(self, states, action, reward, scores):
        self.states = states
        self.action = action
        self.reward = reward
        self.scores = scores

class ReplayBuffer():
    def __init__(self, agent_num, height, width, model, modelpath, game_round):
        #initial the model in the replay buffer
        #model is be defined using pytorch lib

        #the game is the enviroment of route planning game in the project
        self.height = height
        self.width = width
        self.game_round = game_round
        self.agent_numbers = agent_num
        self.model = model
        self.modelpath = modelpath

        self.states = torch.zeros([4, self.agent_numbers, 3, self.height, self.width], dtype=torch.float32)

        #default score setting in the game
        self.acquisition_sum = 700
        self.explored_target_sum = 70
        self.explored_sum = 40
        self.time_decrease = -0.00005
        self.crash_time_penalty = -0.0001
        self.crash_sum = -400

        self.action = torch.tensor([random.randint(0, 4) for i in range(self.agent_numbers)])
        self.memory = []
        self.buffer_limit = 1000

        self.game = None
        self.score = 0
        self.initialGame()

    def collect(self, verbose=1):
        self.loadWeight()
        if self.game.active and self.game.state < self.game_round:
            #observe part of the bot

            self.states[0] = self.states[1]
            self.states[1] = self.states[2]
            self.states[2] = self.states[3]
            s = np.array([self.game.outputAgentImage(i) for i in range(self.agent_numbers)]).astype(np.float32)
            s = torch.from_numpy(s)
            self.states[3] = s

            action = self.select_action()
            self.game.runOneRound([intoCommand(i, action[i]) for i in range(self.agent_numbers)])
            
            score = self.game.outputScore()
            reward = score - self.score
            

            if verbose == 1:
                print(score)

            data = Data(self.states, self.action, reward, self.score)
            self.memory.append(data)

            self.action = action
            self.score = score

            if len(self.memory) > self.buffer_limit:
                self.memory.remove(self.memory[0])
        else:
            #only occur when the game is not active
            self.initialGame()

    def select_action(self):
        if random.random() < RANDOM_THRES:
            return torch.tensor([random.randint(0, 4) for i in range(self.agent_numbers)])
        else:
            with torch.no_grad():
                return self.model(self.states[1:].unsqueeze(0)).max(2)[1].view(self.agent_numbers)

    def save_state(self, path):
        for i in range(self.agent_numbers):
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
            s = np.array([self.game.outputAgentImage(i) for i in range(self.agent_numbers)]).astype(np.float32)
            s = torch.from_numpy(s)
            self.states[3] = s

            action = model(self.states[1:].unsqueeze(0)).max(2)[1].view(self.agent_numbers)
            print(action)
            action = [intoCommand(i, action[i]) for i in range(self.agent_numbers)]
            self.game.runOneRound(action)
            print(self.game.outputScore())
            sleep(0.1)

    def initialGame(self):
        #initial the game envirnment for th replay buffer
        t = (self.height + self.width) / 2

        self.game = Game(self.height, self.width)
        self.game.setRandomMap(self.agent_numbers, int(t * 0.3) ** 2, int(t * 0.1) ** 2)
        self.game.setScore(self.acquisition_sum, self.explored_target_sum, self.explored_sum, self.time_decrease, self.crash_time_penalty, self.crash_sum)

        self.game.runOneRound([Command(i, 0, 0) for i in range(self.agent_numbers)])
        self.score = self.game.outputScore()

        self.states = torch.zeros([4, self.agent_numbers, 3, self.height, self.width], dtype=torch.float32)
        self.loadWeight()
        print('New Game!')

    def loadWeight(self):
        if os.path.exists(self.modelpath):
            self.model.load_state_dict(torch.load(self.modelpath))
        else:
            print('Model weight [{0}] not found'.format(self.modelpath))
        return


def intoCommand(i, command):
    if command == 0: return Command(i, 0, 1)
    elif command == 1: return Command(i, 1, 0)
    elif command == 2: return Command(i, 0, -1)
    elif command == 3: return Command(i, -1, 0)
    return Command(i, 0, 0)