#!/usr/bin/env python
# coding: utf-8

from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product


class TicTacToe():

    def __init__(self):
        # Initialize the board which is a 3x3 squares, represented here as an array.
        # Total 9 squares each initialized as null and can hold a number.
        
        self.state = [np.nan for x in range(0, 9)]
        self.all_possible_numbers = [x for x in range(1, len(self.state) + 1)]
        self.reset()
        
    def is_winning(self, curr_state):
        # Takes state as an input and returns whether any player has won the game or not.
        # Any row (3 consecutive horizontal squares), column (3 consecutive horizontal squares),
        # or 3 consequtive diagonal squares has  sum "15" indicates the game is won, otherwise not.
        
        x = curr_state
        
        row_1 =  x[0] + x[1] + x[2] # Total of digits in row# 1
        row_2 =  x[3] + x[4] + x[5] # Total of digits in row# 2
        row_3 =  x[6] + x[7] + x[8] # Total of digits in row# 3
        col_1 =  x[0] + x[3] + x[6] # Total of digits in column# 1
        col_2 =  x[1] + x[4] + x[7] # Total of digits in column# 2
        col_3 =  x[2] + x[5] + x[8] # Total of digits in column# 3
        diag_1 = x[0] + x[4] + x[8] # Total of digits in diagonal row # 1
        diag_2 = x[2] + x[4] + x[6] # Total of digits in diagonal row # 2
        
        # Variable "is_winnning" is True if any of these totals are 15, else False
        if (row_1 == 15 or row_2 == 15 or row_3 == 15 or col_1 == 15 or col_2 == 15 or col_3 == 15 or diag_1 == 15 or diag_2 == 15):
            is_winning_game = True
        else:
            is_winning_game = False
        
        return is_winning_game
    
    def is_terminal(self, curr_state):
        # Determine wether the game is in terminal state (a player has won  the game or it is tied) or,
        # the game is in non terminal state (in progress)
        
        if self.is_winning(curr_state) == True:
            return True, 'Win'
        elif len(self.allowed_positions(curr_state)) == 0:
            return True, 'Tie'
        else:
            return False, 'Resume'
        
    def allowed_positions(self, curr_state):
        # returns a list of allowed positions for the next move for both - agent and environment.
        
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]
            
    def allowed_values(self, curr_state):
        # returns a list of allowed values for the next move for both - agent and environment.
        
        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)
    
    def action_space(self, curr_state):
        # returns a list of allowed positions & values for the next move, for both - agent and environment
        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)
    
    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        
        x = curr_action[0]
        y = curr_action[1]
        curr_state[x] = y
        return curr_state
    
    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. 
        Hint: First, check the board position after agent's move, whether the game is won/loss/tied. 
        Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""

        # Perform the next move for the agent
        next_state = self.state_transition(curr_state, list(curr_action))
        # Check if the game is in the terminal state
        game_status, game_outcome = self.is_terminal(next_state)
        if (game_status):
            if game_outcome == 'Win':
                # Game is in terminal state and won by agent. Reward for agent is 10
                reward = 10
            else:
                # Game is in terminal state and tied. Reward for agent is 0 points
                reward = 0
        else:
            # Game is not in terminal state. Reward for agent is -1 point
            reward = -1
            pos_list = self.allowed_positions(next_state)
            # Determine environment action. First determine which board positions and values are available.
            # Based on current board position 
            env_values = self.allowed_values(next_state)[1]
            #print('pos_list:',pos_list)
            #print('env_values:', env_values)
            env_action = []
            # Choose a random board position for the environemnt from available free board positions
            env_action.append(np.random.choice(pos_list))
            # Choose a random value for the environment from available values
            env_action.append(np.random.choice(env_values))
            # Perform environment action
            next_state = self.state_transition(next_state, env_action)
        
            # After environment action, check if game is in a terminal state
            game_status, game_outcome = self.is_terminal(next_state)
            if (game_status):
                if game_outcome == 'Win':
                    # Environment has won, reward for agent is -10 points
                    reward = -10
                else:
                    # Game is tied, reward for agent is 0.
                    reward = 0
        return next_state, reward, game_outcome
            
    def reset(self):
        return self.state
