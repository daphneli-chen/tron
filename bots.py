#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""
    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.BOT_NAME = "Patrick's Pet Rock from Spongebob"
        self.order = order
        self.prev_direction = "U"
        self.loc = None
        self.distance_in_dir = 0


    def validNeighbors(self, board, curr, visited):
        ret =[]
        curr_r = curr[0]
        curr_c = curr[1]
        actions = TronProblem.get_safe_actions(board, curr)
        for action in actions:
            if (action == 'U'):
                loc = (curr_r - 1, curr_c)
                if loc not in visited:
                    ret.append((curr_r - 1, curr_c))
            else if (action == 'D'):
                loc = (curr_r + 1, curr_c)
                if loc not in visited:
                    ret.append(loc)
            else if (action == 'L'):
                loc = (curr_r, curr_c - 1)
                if loc not in visited:
                    ret.append(loc)
            else if (action == 'R'):
                loc = (curr_r, curr_c + 1)
                if loc not in visited:
                    ret.append(loc)
        return ret

    def bfs(self, start, board):
        q = []
        dist = {}
        parent = {}
        q.append(start)
        # parent[start] = None
        while len(q) != 0:
            curr = q.pop(0) #gets the first item
            v.add(curr) #marked as visited
            if curr in parent:
                dist[curr] = dist[parent[curr]] + 1
            else:
                dist[curr] = 0 #it was our start point
            for neighbor in self.validNeighbors(board, curr, dist):
                parent[neighbor] = curr
            q.append(self.validNeighbors(board, curr))
        return dist

    def voronoi(self, state):
        """
        """
        num_players = len(state.player_locs)
        me = state.ptm
        #TODO: run bfs for each player, so for each player we have stored a dict where
        #each coordinate's minimum distance is found and compare which player has the
        #shortest distance to go by iterating through each location, your heuristics
        #is determined by how many more space u have :)

        #TODOs: run bfs for each player

        #TODO: iterate through each location

        #TODO: calculate heuristic

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        #TODO: 1) use heuristic to choose best move (plug in left right whatever)
        #2. implement heuristic with a minimax w/ cutoff
        #3. implement alphabeta


        #Future TODO: Learn a heuristic? deep learning? idk add more - dijkstras?
        return "U"

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass


class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
