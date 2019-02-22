# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:15:54 2019

@author: Jalil M
"""

import numpy as np
import sys

from robot import Direction


class HMM:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.matrixT = self.create_matrixT()
        self.none_matrix = self.none_matrix()
        self.f_matrix = self.create_priors()

    def create_priors(self):
        length = self.width * self.height * 4
        priors = [float(1) / length] * length
        return np.array(priors)

    def create_matrixT(self):
        width = self.width
        height = self.height
        result = np.array(np.zeros(shape=(width * height * 4, width * height * 4)))

        for x in range(height):
            for y in range(width):
                for direction in Direction.DIRS:
                    # State at time t-1
                    i = x * height * 4 + y * 4 + direction
                    
                    # Possible states at time t+1
                    poss_trans = self.possible_transitions(x, y, direction)
                    for (px, py, pd), prob in poss_trans:
                        j = px * height * 4 + py * 4 + pd
                        
                        result[i, j] = prob

        return result
    
    def possible_transitions(self, x, y, direction):
        height = self.height
        width = self.width
        neighbors = [(x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y)]
        transitions = []
        
        for x_coord, y_coord in neighbors:
            if not 0 <= x_coord < height or not 0 <= y_coord < width:
                # The neighbor is out of the grid
                pass
            else:
                for poss_direction in Direction.DIRS:
                    if x_coord == 0 and poss_direction == 0: # NORTH
                        continue
                    if x_coord == height - 1 and poss_direction == 2: # SOUTH
                        continue
                    if y_coord == 0 and poss_direction == 3: # WEST
                        continue
                    if y_coord == width - 1 and poss_direction == 1: # EAST
                        continue
                    
                    if poss_direction == direction:
                        prob = 0.7
                    else:
                        wall_directions = sum([x_coord == 0,
                                               x_coord == height - 1,
                                               y_coord == 0,
                                               y_coord == width - 1])
                        if (x_coord == 0 and direction == 0 or
                           x_coord == height - 1 and direction == 2 or
                           y_coord == 0 and direction == 3 or
                           y_coord == width - 1 and direction == 1):
                            # If the robot if facing a wall in this direction
                            prob = 1.0 / (3 - wall_directions)
                        else:
                            prob = 0.3 / (3 - wall_directions)
                        
                    trans = (x_coord, y_coord, poss_direction)
                    transitions.append((trans, prob))
                    
        return transitions

    def assign_adj(self, o, possible_adj, probability):
        for po_x, po_y in possible_adj:
            index = po_x * self.height * 4 + po_y * 4
            for i in range(4):
                o[index + i, index + i] = probability

    def create_sensor_matrix(self, sensed_coord):
        if sensed_coord is None:
            return self.none_matrix
        width = self.width
        height = self.height
        o = np.array(np.zeros(shape=(width * height * 4, width * height * 4)))
        x, y = sensed_coord

        # Assign probability of 0.1 for sensed_coord
        #CHANGED THE MATRIX
        index = x * 4 * height + y * 4
        for i in range(4):
            o[index + i, index + i] = 0.1

        # Assign probability of 0.05 for directly adjacent squares
        self.assign_adj(o, self.possible_adj(x, y), 0.05)
        # Assign probability of 0.025 for directly adjacent squares
        self.assign_adj(o, self.possible_adj2(x, y), 0.025)

        return o

    def none_matrix(self):
        width = self.width
        height = self.height
        o = np.array(np.zeros(shape=(width * height * 4, width * height * 4)))
        for i in range(width * height * 4):
            x = i // (height * 4)
            y = i // 4

            num_adj = 8 - len(self.possible_adj(x, y))
            num_adj2 = 16 - len(self.possible_adj2(x, y))

            o[i, i] = 0.1 + 0.05 * num_adj + 0.025 * num_adj2
        return o

    def possible_adj(self, x, y):
        possible_adj = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y),
                        (x + 1, y + 1)]

        # Bound check
        for po_x, po_y in list(possible_adj):
            if po_x >= self.width or po_x < 0 or po_y >= self.height or po_y < 0:
                possible_adj.remove((po_x, po_y))

        return possible_adj

    def possible_adj2(self, x, y):
        possible_adj2 = [(x - 2, y - 2), (x - 2, y - 1), (x - 2, y), (x - 2, y + 1), (x - 2, y + 2), (x - 1, y - 2),
                         (x - 1, y + 2), (x, y - 2), (x, y + 2), (x + 1, y - 2), (x + 1, y + 2), (x + 2, y - 2),
                         (x + 2, y - 1),
                         (x + 2, y), (x + 2, y + 1), (x + 2, y + 2)]

        # Bound check
        for po_x, po_y in list(possible_adj2):
            if po_x >= self.width or po_x < 0 or po_y >= self.height or po_y < 0:
                possible_adj2.remove((po_x, po_y))

        return possible_adj2

    def forward(self, coord):
        f = self.f_matrix
        o = self.create_sensor_matrix(coord)
        t = self.matrixT
        
        f = t.dot(f).dot(o)
        f /= np.sum(f)

        self.f_matrix = f

    def most_probable(self):
        f = self.f_matrix
        max_prob_idx = np.argmax(f)
        x = max_prob_idx // (self.height * 4)
        y = (max_prob_idx // 4) % self.height
        return (x, y), f[max_prob_idx]
