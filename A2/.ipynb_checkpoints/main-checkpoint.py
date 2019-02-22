# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:39:43 2019

@author: Jalil M
"""

from time import sleep
import argparse

from grid import Grid
from hmm import HMM
from robot import Robot, Sensor

def start_robot(size):
    """
    Creating the World and robot and loops over probabilities.
    """
    grid = Grid(size.width, size.height)
    sensor = Sensor(grid)
    hmm = HMM(size.width, size.height)
    robot = Robot(sensor, hmm)
    moves = 0
    guessed_r = 0
    while True:
        grid.move_robot()
        moves += 1
        print ("\nRobot position: ", grid.robot_location)
        guessed, probability = robot.guess_move()
        if guessed == grid.robot_location:
            guessed_r += 1
        man_distance = abs(guessed[0] - grid.robot_location[0]) + abs(guessed[1] - grid.robot_location[1])
        print ("Manhattan distance: ", man_distance)
        print ("Robot has an accuracy of:", float(guessed_r) / moves, "during this time.")
        sleep(1)

if __name__ == '__main__':
    """
    USAGE :
    > python main.py --width 10 --height 10
    The arguments define respectively the width and height of the grid.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    if args.width <= 1 or args.height <= 1:
        raise argparse.ArgumentTypeError("Values have to be >1")
    
    start_robot(args)

