"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
###################这里要改##################
UNIT = 40   # pixels
MAZE_H = 8  # grid height
MAZE_W = 8  # grid width
global j
j = 0
###################这里要改##################

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        # build hells
        ###################这里要改##################
        self.hells = []
        def build_hells(numbers, x, y, who_move):
            global j
            j = j+numbers
            for i in range(0, numbers):
                if who_move == 'x':
                    hell_center = origin + np.array([UNIT * (x + i), UNIT * y])
                    s = self.canvas.coords(self.canvas.create_rectangle(
                        hell_center[0] - 15, hell_center[1] - 15,
                        hell_center[0] + 15, hell_center[1] + 15,
                        fill='black'))
                    self.hells.append(s)
                if who_move == 'y':
                    hell_center = origin + np.array([UNIT * x, UNIT * (y + i)])
                    s =self.canvas.coords(self.canvas.create_rectangle(
                        hell_center[0] - 15, hell_center[1] - 15,
                        hell_center[0] + 15, hell_center[1] + 15,
                        fill='black'))
                    self.hells.append(s)


        # build_hells(10, 0, 3, 'x')
        # build_hells(6, 9, 5, 'x')
        # build_hells(4, 3, 6, 'y')
        # build_hells(4, 3, 9, 'x')
        # build_hells(4, 7, 6, 'y')
        # build_hells(3, 12, 9, 'x')
        # build_hells(4, 10, 11, 'y')
        build_hells(4,1,1,'x')
        build_hells(5, 6, 2, 'y')
        build_hells(4, 1, 3, 'y')
        build_hells(3, 2, 6, 'x')
        build_hells(1,7,5,'x')
        ###################这里要改##################

        # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        oval_center = origin + np.array([UNIT * 7, UNIT * 7])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        # time.sleep(0.4)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        # time.sleep(0.01)
        s = self.canvas.coords(self.rect)

        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        ###########################将hells改为墙壁##############################
        x = self.canvas.coords(self.rect)
        x[0] = s[0]+base_action[0]
        x[1] = s[1]+base_action[1]

        for i in range(0,j):
            if (x[0] ==self.hells[i][0]) & (x[1] ==self.hells[i][1]):
                XXX = True
                break
            else:
                XXX = False
        if XXX:
            pass
        else:
            self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        ###########################将hells改为墙壁##########################################

        # self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state
        ###################这里要改##################
        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
         ###########################将hells改为墙壁##############################
        # elif next_coords in self.hells:
        #     #reward = -1
        #     #done = True
        #     pass
        ###########################将hells改为墙壁##############################
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    ###################这里要改##################
    def render(self):
        # time.sleep(0.01)
        self.update()