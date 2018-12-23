import os
import sys
import datetime
import numpy as np
import torch
from utils import *
from model import *
from lib.game.game import Game, Command
from lib.game.agent import Agent
#--------------------------------------------------------------------------------
#acting.py is the class for collect the training data from the game
#--------------------------------------------------------------------------------
class ReplayBuffer():
    def __init__(self, agent_num, model_structure, model_weight, height, width, path = "./buffer", cuda=False):
        #initial the model in the replay buffer
        #model is be defined using pytorch lib
        self.model = model_structure
        self.model_weight = model_weight

        self.loadWeight()

        #the game is the enviroment of route planning game in the project
        self.height = height
        self.width = width

        self.game = None

        self.hx = None
        self.cx = None
        self.bot_observe = None

        # self.agent_numbers = int(((self.height + self.width) / 2) * 0.3)
        self.agent_numbers = agent_num

        #default score setting in the game
        self.acquisition_sum = 700
        self.explored_target_sum = 70
        self.explored_sum = 40
        self.time_decrease = -0.00005
        self.crash_time_penalty = -0.0001
        self.crash_sum = -20

        #path is the folder of the replay buffer
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        self.IOlimit = 20
        self.buffer_limit = 1000

        self.cuda = cuda
        print(cuda)

    def collect(self, game_round):
        #data is stored in the python list using pickle
        #data list is a 2d list index is the batch index of collection
        if self.game == None:
            self.initialGame()
        
        if self.cuda:
            self.model.cuda()
        #collecting part
        while True:
            data = []

            collect_iter = 0

            while collect_iter < self.IOlimit:
                temp = []
                if self.game.active and self.game.state < game_round:
                    #observe part of the bot
                    # bot_observe = np.zeros((self.agent_numbers, 3, self.width, self.height))

                    # for i in range(self.agent_numbers):
                    #     bot_observe[i] = self.game.outputAgentImage(i)

                    if self.bot_observe is None:
                        self.bot_observe = [self.game.outputAgentImage(i) for i in range(self.agent_numbers)]
                        self.bot_observe = np.array(bot_observe).astype(np.uint8)

                    # bot_observe = np.uint8(bot_observe)
                    #batch * width * height * 3 (numpy array)[0]
                    temp.append(self.bot_observe)

                    temp.append(self.hx) #batch * 512[1](tensor)[1]
                    temp.append(self.cx) #batch * 512[1](tensor)[2]

                    #model output process
                    self.bot_observe = torch.from_numpy(self.bot_observe).float()

                    if self.cuda:
                        self.hx = self.hx.cuda()
                        self.cx = self.cx.cuda()
                        self.bot_observe = self.bot_observe.cuda()

                    critic_score, bot_command, (self.hx, self.cx) = self.model(self.bot_observe, self.hx, self.cx)

                    #bot action tensor
                    temp.append(bot_command.cpu()) #tensor[3]

                    #game score output
                    temp.append(self.game.outputScore()) #float[4]

                    #input the commands of the model to the game
                    commands = self.interpretAction(bot_command.cpu())
                    self.game.runOneRound(commands)

                    self.bot_observe = [self.game.outputAgentImage(i) for i in range(self.agent_numbers)]
                    self.bot_observe = np.array(self.bot_observe).astype(np.uint8)

                    #batch * width * height * 3 (numpy array)[5]
                    temp.append(self.bot_observe)

                    temp.append(self.hx.cpu()) #batch * 512[1](tensor)[6]
                    temp.append(self.cx.cpu()) #batch * 512[1](tensor)[7]

                    #game score output
                    temp.append(self.game.outputScore()) #float[8]

                    data.append(temp)

                    collect_iter += 1

                    if collect_iter % 5 == 0 and collect_iter is not 0:
                        print('Collecting Progress: ', collect_iter, ' / ', self.IOlimit, ' | Score: %02.5f' % self.game.outputScore())
                else:
                    #only occur when the game is not active
                    self.initialGame()

            #print('Collecting process done\nStart writing file...')
            
            for i in range(self.IOlimit):
                filename = self.path + '/' + datetime.datetime.now().isoformat() + '.pkl'
                save_object(filename, data[i])

            #print('All collection process done\n')

            del data
            self.loadWeight()

    def loadWeight(self):
        if os.path.exists(self.model_weight):
            self.model.load_state_dict(torch.load(self.model_weight))
        else:
            print('Model weight [{0}] not found'.format(self.model_weight))
        return

    def initialGame(self):
        #initial the game envirnment for th replay buffer
        t = (self.height + self.width) / 2

        self.game = Game(self.height, self.width)
        self.game.setRandomMap(self.agent_numbers, int(t * 0.3) ** 2, 1)
        self.game.setScore(self.acquisition_sum, self.explored_target_sum, self.explored_sum, self.time_decrease, self.crash_time_penalty, self.crash_sum)

        commands = [Command(i, 0, 0) for i in range(self.agent_numbers)]
        self.game.runOneRound(commands)

        #the memory tensor of the model in the lstm
        self.hx = torch.zeros([self.agent_numbers, 512], dtype=torch.float32)
        self.cx = torch.zeros([self.agent_numbers, 512], dtype=torch.float32)

        self.bot_observe = [self.game.outputAgentImage(i) for i in range(self.agent_numbers)]
        self.bot_observe = np.array(self.bot_observe).astype(np.uint8)

        print('New Game!')

    def resetGameScore(self, acquisition_sum, explored_target_sum, explored_sum, time_decrease, crash_time_penalty, crash_sum):
        #the score calculating standard of the game
        self.acquisition_sum = acquisition_sum
        self.explored_target_sum = explored_target_sum
        self.explored_sum = explored_sum
        self.time_decrease = time_decrease
        self.crash_time_penalty = crash_time_penalty
        self.crash_sum = crash_sum

    def countFileNum(self, path):
        file_list = os.listdir(path)
        count = 0
        for i in range(len(file_list)):
            if file_list[i].endswith('.pkl'):
                count += 1

        return count

    def grabFileName(self, path):
        file_list = os.listdir(path)
        name_list = []
        for i in range(len(file_list)):
            if file_list[i].endswith('.pkl'):
                name_list.append(file_list[i])

        return name_list

    def resetIOBatch(self, new_limit):
        self.IOlimit = new_limit

    def resetBufferLimit(self, new_limit):
        self.buffer_limit = new_limit

    def interpretAction(self, command_tensor):
        command_tensor = command_tensor.view(-1, 5)
        command = torch.max(command_tensor, 1)[1].tolist()
        commands = [self.intoCommand(i, command[i]) for i in range(len(command))]
        return commands

    def intoCommand(self, i, command):
        if command == 0: return Command(i, 0, 1)
        elif command == 1: return Command(i, 1, 0)
        elif command == 2: return Command(i, 0, -1)
        elif command == 3: return Command(i, -1, 0)
        return Command(i, 0, 0)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: python3 RPbuffer_v2.py [model path] [height] [weight] [cuda]')
        exit()

    model_weight = sys.argv[1]
    HEIGHT = int(sys.argv[2])
    WIDTH = int(sys.argv[3])
    cuda = False

    if sys.argv[4] != '-1':
        cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]

    replayBuffer = ReplayBuffer(5, ActorCritic(HEIGHT, WIDTH), model_weight, HEIGHT, WIDTH, cuda=cuda)

    replayBuffer.collect(512)
