import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.game.game import Game, Command
from lib.game.agent import Agent

def randomGame():
    print("\n== init random game ==")
    game = Game(20, 20)  # height, width
    game.setRandomMap(3, 3, 4)  # numbers of agents, targets, obstacles
    game.setScore(100, 20, 10, -2, -0.04, -20)

    game.printGodInfo()

    print("\n== 1st round ==")
    commands = []
    commands.append(Command(0, 1, 1))  # id, dx, dy
    commands.append(Command(1, -1, 1))
    commands.append(Command(2, 1, -1))

    game.runOneRound(commands)

    game.printConsoleInfo()
    print("Score: " + str(game.outputScore()))

    print("\n== 2ed round ==")
    commands = []
    commands.append(Command(0, 1, 1))
    commands.append(Command(1, -1, 1))
    commands.append(Command(2, 1, -1))

    game.runOneRound(commands)

    game.printConsoleInfo()
    print("Score: " + str(game.outputScore()))

def manualGame():
    print("\n== init manual setting game ==")
    height = 20
    width = 20
    game = Game(height, width)

    obstacles = [
        {"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3}
    ]
    targets = [
        {"x": 10, "y": 10}, {"x": 12, "y": 12}, {"x": 13, "y": 13}
    ]
    game.setObstacles(obstacles)
    game.setTargets(targets)

    agents = {
        0: Agent(0, 0, 0, height, width),  # id, x, y, height, width
        1: Agent(1, 14, 14, height, width),
        2: Agent(2, 15, 15, height, width),
    }
    game.setAgents(agents)
    game.setScore(100, 20, 10, -0.0005, 0, -20)

    game.printGodInfo()
    print("Score: " + str(game.outputScore()))

    print("\n== 1st round ==")
    commands = []
    commands.append(Command(0, 0, 1))  # id, dx, dy
    commands.append(Command(1, -1, 1))
    commands.append(Command(2, 1, -1))

    game.runOneRound(commands)

    game.printConsoleInfo()
    print("Score: " + str(game.outputScore()))

    print("\n== 2ed round ==")
    commands = []
    commands.append(Command(0, 1, 1))
    commands.append(Command(1, -1, 1))
    commands.append(Command(2, 1, -1))

    game.runOneRound(commands)

    game.printConsoleInfo()
    print("Score: " + str(game.outputScore()))

    print("\n== 3ed round ==")
    commands = []
    commands.append(Command(0, 1, 1))
    commands.append(Command(1, -1, 1))
    commands.append(Command(2, -1, 0))

    game.runOneRound(commands)

    game.printConsoleInfo()
    print("Score: " + str(game.outputScore()))


#randomGame()
manualGame()
