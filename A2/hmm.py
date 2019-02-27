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

        for x in range(width):
            for y in range(height):
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
        neighbors = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        transitions = []
        
        can_go_forward = True
        
        x_coord, y_coord = neighbors[direction]
        if not 0 <= y_coord < height or not 0 <= x_coord < width:
            can_go_forward = False
        
        for d, (x_coord, y_coord) in enumerate(neighbors):
            if not 0 <= y_coord < height or not 0 <= x_coord < width:
                # The neighbor is out of the grid
                continue
            else:
                if d == direction:
                    transitions.append(((x_coord, y_coord, d), 0.7))
                else:
                    if can_go_forward:
                        p = 0.3
                    else:
                        p = 1.0
                    transitions.append(((x_coord, y_coord, d), p))
                    
        if can_go_forward:
            transitions = [(pos, p / (len(transitions) - 1)) if p != 0.7 else (pos, p) for pos, p in transitions]
        else:
            transitions = [(pos, p / len(transitions)) for pos, p in transitions]
                    
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

        # prob of 0.1
        index = x * 4 * height + y * 4
        for i in range(4):
            o[index + i, index + i] = 0.1

        # prob 0.05
        self.assign_adj(o, self.possible_adj(x, y), 0.05)
        # prob 0.025
        self.assign_adj(o, self.possible_adj2(x, y), 0.025)

        return o

    def none_matrix(self):
        width = self.width
        height = self.height
        o = np.array(np.zeros(shape=(width * height * 4, width * height * 4)))
        for i in range(width * height * 4):
            x = (i // 4) // height
            y = (i // 4) % self.height

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
        x = (max_prob_idx // 4) // self.height
        y = (max_prob_idx // 4) % self.height
        return (x, y), f[max_prob_idx]
