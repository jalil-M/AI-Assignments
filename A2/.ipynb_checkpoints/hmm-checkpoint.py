# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:15:54 2019

@author: Jalil M
"""

import numpy as np

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

        for i in range(width * height * 4):
            x = i / (height * 4)
            y = (i / 4) % height
            heading = i % 4
            prev_states = self.probable_trans((x, y, heading))
            #CHANGED THE MATRIX
            for (xcoord, ycoord, direction), probability in prev_states:
                result[i, int(xcoord * 4 + ycoord * 4 + direction)] = probability
        return result

    def probable_trans(self, state):
        x, y, direction = state
        # came from: NORTH, EAST, SOUTH, WEST
        neighbors = [(x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y)]
        prev_square = neighbors[direction]
        prev_x, prev_y = prev_square

        # Check bounds
        if prev_x < 0 or prev_x >= self.width or prev_y < 0 or prev_y >= self.height:
            return []

        # Always 0.7 chance if coming in same direction.
        square_dir = [((prev_x, prev_y, direction), 0.7)]
        dirs_left = list(Direction.DIRS)
        dirs_left.remove(direction)
        # Check if any directions point to walls.
        faces_wall = []
        if Direction.WEST in dirs_left:
            if prev_x == 0:
                faces_wall.append((prev_x, prev_y, Direction.WEST))
            else:
                square_dir.append(((prev_x, prev_y, Direction.WEST), 0.1))
        if Direction.EAST in dirs_left:
            if prev_x == self.width - 1:
                faces_wall.append((prev_x, prev_y, Direction.EAST))
            else:
                square_dir.append(((prev_x, prev_y, Direction.EAST), 0.1))
        if Direction.SOUTH in dirs_left:
            if prev_y == 0:
                faces_wall.append((prev_x, prev_y, Direction.SOUTH))
            else:
                square_dir.append(((prev_x, prev_y, Direction.SOUTH), 0.1))
        if Direction.NORTH in dirs_left:
            if prev_y == self.height - 1:
                faces_wall.append((prev_x, prev_y, Direction.NORTH))
            else:
                square_dir.append(((prev_x, prev_y, Direction.NORTH), 0.1))

        for state in faces_wall:
            square_dir.append((state, float(1) / (4 - len(faces_wall))))
        return square_dir

    def assign_adj(self, o, possible_adj2, probability):
        for po_x, po_y in possible_adj2:
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
        index = x * 4 + y * 4
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
            x = i / (height * 4)
            y = (i / 4) % height

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
        print("WE PRINT THE VALUE : ", np.sum(f))
        f /= np.sum(f)

        self.f_matrix = f

    def most_probable(self):
        f = self.f_matrix
        max_prob_idx = np.argmax(f)
        x = max_prob_idx / (self.height * 4)
        y = (max_prob_idx / 4) % self.height
        return (x, y), f[max_prob_idx]
