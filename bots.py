#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from collections import deque
import time

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


    def validNeighbors(self, board, curr, my_visited, o_visited, q):
        curr_r = curr[0]
        curr_c = curr[1]
        actions = TronProblem.get_safe_actions(board, curr)
        for action in actions:
            if (action == 'U'):
                loc = (curr_r - 1, curr_c)
                if loc not in my_visited and loc not in o_visited:
                    q.put((curr_r - 1, curr_c))
            elif (action == 'D'):
                loc = (curr_r + 1, curr_c)
                if loc not in my_visited and loc not in o_visited:
                    q.put(loc)
            elif (action == 'L'):
                loc = (curr_r, curr_c - 1)
                if loc not in my_visited and loc not in o_visited:
                    q.put(loc)
            elif (action == 'R'):
                loc = (curr_r, curr_c + 1)
                if loc not in my_visited and loc not in o_visited:
                    q.put(loc)
        return q

    def getNextLoc(self, curr, action):
        curr_r = curr[0]
        curr_c = curr[1]
        if (action == 'U'):
            return (curr_r - 1, curr_c)
        elif (action == 'D'):
            return (curr_r + 1, curr_c)
        elif (action == 'L'):
            return (curr_r, curr_c - 1)
        elif (action == 'R'):
            return  (curr_r, curr_c + 1)
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

    # def bfs(self, start, board):
    #     """
    #     start: a tuple representing the player's location
    #     board: the board of the game
    #     """
    #     q = []
    #     # dist = {}
    #     dist = np.ones((len(board), len(board[0]))) * (float("inf"))
    #     dist[start[0]][start[1]] = 0
    #     seen = {}
    #     #uncomment parent dict if you want to know which position led to best transition
    #     # parent = {}
    #     q.append(start)
    #     seen[start] = 1
    #     # parent[start] = None
    #     while len(q) != 0:
    #         curr = q.pop(0) #gets the first item
    #         seen[curr] = 1
    #         # if curr in parent:
    #         #     dist[curr] = dist[parent[curr]] + 1
    #         # else:
    #         #     dist[curr] = 0 #it was our start point
    #         for neighbor in self.validNeighbors(board, curr, seen):
    #             temp_cost = dist[curr[0]][curr[1]] + 1
    #             if temp_cost < dist[neighbor[0]][neighbor[1]]:
    #                 dist[neighbor[0]][neighbor[1]] = temp_cost
    #                 # parent[neighbor] = curr
    #                 # q.append(neighbor)
    #                 q = q + self.validNeighbors(board, curr, seen)
    #
    #     return dist

    def bds(self, asp, state, ptm):
        """
        Does a bidirectional search from each player out, finishes when all squares have been
        visited
        """
        board = state.board
        separated = True
        my_start = state.player_locs[0] #assuming player is 0 for my
        o_start = state.player_locs[1] #1 for other
        my_frontier = deque()
        my_visited = set()
        if my_start != None:
            my_frontier.append((my_start))
            my_visited.add((my_start))


        o_frontier = deque()
        o_visited = set()
        if o_start != None:
            o_frontier.append((o_start))
            o_visited.add((o_start))

        while my_frontier or o_frontier:
            temp_front = deque()
            while my_frontier:
                curr = my_frontier.popleft()
                for action in asp.get_safe_actions(board, curr):
                    new_loc = asp.move(curr, action)
                    if new_loc not in my_visited and new_loc not in o_visited:
                        my_visited.add((new_loc))
                        temp_front.append((new_loc))
                    elif new_loc in o_visited:
                        separated = False
            my_frontier = temp_front #new depth level of neighbors
            temp_front = deque()
            while o_frontier:
                curr = o_frontier.popleft()
                for action in asp.get_safe_actions(board, curr):
                    new_loc = asp.move(curr, action)
                    if new_loc not in my_visited and new_loc not in o_visited:
                        o_visited.add((new_loc))
                        temp_front.append((new_loc))
                    elif new_loc in my_visited:
                        separated = False
            o_frontier = temp_front
        print(len(my_visited))
        diff = len(my_visited) - len(o_visited)
        if ptm == 0:
            return diff, separated
        else:
            return -1 *diff, separated



    def voronoi(self, asp, state, startingPlayer):
        """
        asp: a tron problem
        state: current state of the tron problem

        returns difference
        """
        return self.bds(asp, state, startingPlayer)[0]

    def wallDecide(self, asp):
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
        me = start_state.player_to_move()
        possibleActions = asp.get_safe_actions(start_state.board, start_state.player_locs[me])
        actionBest = "U"
        #
        # if self.bds(asp, start_state, me)[1]:
        #     print("separated")
        #     return self.wallDecide(asp)

        bestVal = float("-inf")
        depth = 1
        alpha = float("-inf")

        #3. implement alphabeta
        print(start_state)
        print(possibleActions)
        for action in possibleActions:
            newState = asp.transition(start_state, action)
            receivedVal = self.abCutMin(asp, newState, alpha, float("inf"), 6, depth, me)
            print("received Val for action", action, "is", receivedVal)
            if receivedVal > bestVal:
                bestVal = receivedVal
                actionBest = action
            alpha = max(receivedVal, alpha)
        #Future TODO: Learn a heuristic? deep learning? idk add more - dijkstras?
        print("my action:")
        print(actionBest)
        print("best value:")
        print(bestVal)
        return actionBest


    def abCutMax(self, asp, state, alpha, beta, cutoff, depth, actingPlayer):
        if asp.is_terminal_state(state):
            print("actingPlayer is", actingPlayer)

            print("in terminal state for abcutmax", asp.evaluate_state(state)[0], asp.evaluate_state(state)[1])
            if asp.evaluate_state(state)[actingPlayer] == 1:
                return 1000 * abs(self.voronoi(asp, state, actingPlayer))
            else:
                print("i lost and",  self.voronoi(asp, state, actingPlayer))
                return -1 * 1000 * abs(self.voronoi(asp, state, actingPlayer))
            # return 1000 * asp.evaluate_state(state)[actingPlayer]
        if depth >= cutoff:
            return self.voronoi(asp, state, state.ptm)

        value = float("-inf")
        possibleActions = asp.get_safe_actions(state.board, state.player_locs[actingPlayer])
        if not possibleActions:
            print("how did we get here")
            return -1 * 1000 * abs(self.voronoi(asp, state, actingPlayer))
        for actions in possibleActions:
            value = max(value, self.abCutMin(asp, asp.transition(state, actions), alpha, beta, cutoff, depth+1, actingPlayer))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def abCutMin(self, asp, state, alpha, beta, cutoff, depth, actingPlayer):
        if asp.is_terminal_state(state):
            print(self.voronoi(asp, state, actingPlayer))
            print("actingPlayer is", actingPlayer)
            print("in terminal state for abcutmin", asp.evaluate_state(state)[0], asp.evaluate_state(state)[1])
            if asp.evaluate_state(state)[actingPlayer] == 1:
                return 1000 * abs(self.voronoi(asp, state, actingPlayer))
            else:
                print("i lost and",  -self.voronoi(asp, state, actingPlayer))
                return -1000 * abs(self.voronoi(asp, state, actingPlayer))
            # return 1000 * asp.evaluate_state(state)[actingPlayer]
        if depth >= cutoff:
            other = 0
            if actingPlayer == 0:
                other = 1
            return self.voronoi(asp, state, state.ptm)

        value = float("inf")

        possibleActions = asp.get_safe_actions(state.board, state.player_locs[actingPlayer])
        if not possibleActions:
            return -1 * 1000 * abs(self.voronoi(asp, state, actingPlayer))
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
