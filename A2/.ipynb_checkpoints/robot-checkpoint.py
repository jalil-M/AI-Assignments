# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:39:43 2019

@author: Jalil M
"""

import random 

class Sensor:
    def __init__(self, grid):
        self.grid = grid
    p = 0.1
    p_s = 0.05
    p_s2 = 0.025

    def sense_location(self):
        rand = random.random()
        if rand <= self.p:
            return self.grid.robot_location
        elif rand <= self.p + self.p_s * 8:
            return self.grid.robot_adj_location()
        elif rand <= self.p + self.p_s * 8 + self.p_s2 * 16:
            return self.grid.robot_adj2_location()
        else:
            return None

class Robot:
    def __init__(self, sensor, hmm):
        self.sensor = sensor
        self.x_guess, self.y_guess = 0, 0
        self.hmm = hmm

    def guess_move(self):
        sensed = self.sensor.sense_location()
        print("Sensor thinks: ", sensed)
        self.hmm.forward(sensed)
        guessed, probability = self.hmm.most_probable()
        print("Robot predicts it is in: ", guessed, " with probability: ", probability)
        return guessed, probability


class Direction:
    NORTH, EAST, SOUTH, WEST = range(4)
    DIRS = [NORTH, EAST, SOUTH, WEST]

    def __init__(self):
        pass

    @classmethod
    def random(cls, exempt_dir=None):
        dirs = [cls.NORTH, cls.EAST, cls.SOUTH, cls.WEST]
        if exempt_dir:
            dirs.remove(exempt_dir)
            return dirs[random.randint(0, 2)]
        else:
            return dirs[random.randint(0, 3)]
