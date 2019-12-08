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
            elif (action == 'D'):
                loc = (curr_r + 1, curr_c)
                if loc not in visited:
                    ret.append(loc)
            elif (action == 'L'):
                loc = (curr_r, curr_c - 1)
                if loc not in visited:
                    ret.append(loc)
            elif (action == 'R'):
                loc = (curr_r, curr_c + 1)
                if loc not in visited:
                    ret.append(loc)
        return ret

    # def bfs(self, start, board):
    #     """
    #     start: a tuple representing the player's location
    #     board: the board of the game
    #     """
    #     q = []
    #     dist = {}
    #
    #     parent = {}
    #     q.append(start)
    #     # parent[start] = None
    #     while len(q) != 0:
    #         curr = q.pop(0) #gets the first item
    #         if curr in parent:
    #             dist[curr] = dist[parent[curr]] + 1
    #         else:
    #             dist[curr] = 0 #it was our start point
    #         for neighbor in self.validNeighbors(board, curr, dist):
    #             parent[neighbor] = curr
    #         q = q + self.validNeighbors(board, curr, dist)
    #     return dist

    def bfs(self, start, board):
        """
        start: a tuple representing the player's location
        board: the board of the game
        """
        q = []
        # dist = {}
        dist = np.ones((len(board), len(board[0]))) * (float("inf"))
        dist[start[0]][start[1]] = 0
        seen = {}
        #uncomment parent dict if you want to know which position led to best transition
        # parent = {}
        q.append(start)
        seen[start] = 1
        # parent[start] = None
        while len(q) != 0:
            curr = q.pop(0) #gets the first item
            seen[curr] = 1
            # if curr in parent:
            #     dist[curr] = dist[parent[curr]] + 1
            # else:
            #     dist[curr] = 0 #it was our start point
            for neighbor in self.validNeighbors(board, curr, seen):
                temp_cost = dist[curr[0]][curr[1]] + 1
                if temp_cost < dist[neighbor[0]][neighbor[1]]:
                    dist[neighbor[0]][neighbor[1]] = temp_cost
                    # parent[neighbor] = curr
                    # q.append(neighbor)
                    q = q + self.validNeighbors(board, curr, seen)

        return dist

    def voronoi(self, asp, state):
        """
        asp: a tron problem
        state: current state of the tron problem

        returns difference
        """
        num_players = len(state.player_locs)
        me = state.ptm
        board = state.board
        other = 0
        if me == 0:
            other = 1
        #TODO: run bfs for each player, so for each player we have stored a dict where
        #each coordinate's minimum distance is found and compare which player has the
        #shortest distance to go by iterating through each location, your heuristics
        #is determined by how many more space u have :)

        if asp.is_terminal_state(state):
            return float("-inf")
        my_dist = self.bfs(state.player_locs[me], board)
        other_dist = self.bfs(state.player_locs[other], board)

        my_count = np.sum(my_dist < other_dist)
        other_count = np.sum(my_dist > other_dist)

        return my_count - other_count

    # def voronoi(self, asp, state):
    #     """
    #     asp: a tron problem
    #     state: current state of the tron problem
    #
    #     returns difference
    #     """
    #     num_players = len(state.player_locs)
    #     me = state.ptm
    #     board = state.board
    #     other = 0
    #     if me == 0:
    #         other = 1
    #     #TODO: run bfs for each player, so for each player we have stored a dict where
    #     #each coordinate's minimum distance is found and compare which player has the
    #     #shortest distance to go by iterating through each location, your heuristics
    #     #is determined by how many more space u have :)
    #     # player_dicts = {}
    #     # num_players = len(asp.player_locs)
    #     # for p in range(num_players):
    #     #     player_dicts[p] = self.bfs(asp.player_locs[p])
    #     # player_counts = np.zeros(num_players)
    #     #
    #     # for r in range(len(board)):
    #     #     for c in range(len(board[0])):
    #     #         best_player = 0
    #     #         min_moves = int(float("inf"))
    #     #         for p in range(num_players):
    #     #             curr_dict = player_dicts[p]
    #     #             if curr_dict[(r,c)] < min_moves:
    #     #                 best_player = p
    #     #                 min_moves = curr_dict[(r,c)]
    #     #             if curr_dict[(r,c)] == min_moves:
    #     #
    #     #         player_counts[best_player] = player_counts[best_player] + 1
    #     # difference = player_counts[0]
    #     # for i in range(1, num_players)
    #
    #     my_dict = self.bfs(state.player_locs[me], board)
    #     other_dict = self.bfs(state.player_locs[other], board)
    #
    #     my_count = 0
    #     other_count = 0
    #     for r in range(len(board)):
    #         for c in range(len(board[0])):
    #             if (r,c) not in my_dict and (r,c) not in other_dict:
    #                 continue
    #             elif (r,c) not in my_dict:
    #                 other_count +=1
    #                 continue
    #             elif (r,c) not in other_dict:
    #                 my_count += 1
    #                 continue
    #             if other_dict[(r,c)] < my_dict[(r,c)]:
    #                 other_count += 1
    #             elif other_dict[(r,c)] != my_dict[(r,c)]:
    #                 my_count += 1
    #     return my_count - other_count

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        #TODO: 1) use heuristic to choose best move (plug in left right whatever)
        #2. implement heuristic with a minimax w/ cutoff
        start_state = asp.get_start_state()
        possibleActions = asp.get_available_actions(start_state)
        me = start_state.player_to_move()
        actionBest = "U"

        bestVal = float("-inf")
        depth = 1
        alpha = float("-inf")

        #3. implement alphabeta
        for action in possibleActions:
            newState = asp.transition(start_state, action)
            receivedVal = self.abCutMin(asp, newState, alpha, float("inf"), 3, depth, me)
            if receivedVal > bestVal:
                bestVal = receivedVal
                actionBest = action
            alpha = max(receivedVal, alpha)
        #Future TODO: Learn a heuristic? deep learning? idk add more - dijkstras?
        return actionBest


    def abCutMax(self, asp, state, alpha, beta, cutoff, depth, actingPlayer):
        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[actingPlayer]
        if depth >= cutoff:
            return self.voronoi(asp, state)

        value = float("-inf")
        for actions in asp.get_available_actions(state):
            value = max(value, self.abCutMin(asp, asp.transition(state, actions), alpha, beta, cutoff, depth+1, actingPlayer))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def abCutMin(self, asp, state, alpha, beta, cutoff, depth, actingPlayer):
        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[actingPlayer]
        if depth >= cutoff:
            return self.voronoi(asp, state)

        value = float("inf")
        for actions in asp.get_available_actions(state):
            value = min(value, self.abCutMax(asp, asp.transition(state, actions), alpha, beta, cutoff, depth+1, actingPlayer))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

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
