from typing import get_args
import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["kr8753bl-s"] # TODO: fill this list with your stil-id's
AI = 1
OPPONENT = -1

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

def student_move(state):
   """
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """

   return alpha_beta(state, 5)

def alpha_beta(state, depth):

   moves = {}

   for child in get_children(state, AI):
      v = min_value(child[0], depth - 1, -np.inf, np.inf)
      moves[child[1]] = v

   move = max(moves, key=moves.get)

   return move


def min_value(state, depth, alpha, beta):

   if has_won(state, OPPONENT):
      return -np.inf

   if depth == 0:
      return eval_move(state)

   v = np.inf

   for child in get_children(state, OPPONENT):
      v = min(v, max_value(child[0], depth - 1, alpha, beta))
      if v <= alpha:
         return v
      beta = min(beta, v)

   return v


def max_value(state, depth, alpha, beta):

   if has_won(state, AI):
      return np.inf

   if depth == 0:
      return eval_move(state) 

   v = -np.inf 

   for child in get_children(state, AI):
      v = max(v, min_value(child[0], depth - 1, alpha, beta))
      if beta <= v:
         return v
      alpha = max(alpha, v)

   return v


def has_won(state, player):
   """ 
   returns true if player has four in a row 
   """

   # Test rows
   for row in range(len(state)):
      for col in range(len(state[row]) - 3):
         if sum(state[row][col:col + 4]) == 4*player:
            return True

   # Test columns on transpose array
   transposed_board = np.transpose(state)
   for row in range(len(state)):
      for col in range(len(state[row]) - 3):
         if sum(transposed_board[row][col:col + 4]) == 4*player:
            return True

   # Test diagonal \
   for row in range(len(state) - 3):
      for col in range(len(state[row]) - 3):
         value = 0
         for k in range(4):
            value += state[row + k][col + k]
            if value == 4*player:
               return True

   # Test reverse diagonal /
   for row in range(len(state) - 3):
      for col in range(3, len(state[row])):
         value = 0
         for k in range(4):
            value += state[row + k][col - k]
            if value == 4*player:
               return True

   return False


def eval_move(state):
   """ 
   calculates the total score of all horizontal, vertical and diagonal windows of length 4 (-1 for oppopnent +1 for AI)
   """

   utility = 0
   # horizontal score
   for row in range(len(state)):
      for col in range(len(state[row]) - 3):
         utility += sum(state[row][col:col + 4])

   # vertical score
   transposed_board = np.transpose(state)
   for row in range(len(state)):
      for col in range(len(state[row]) - 3):
         utility += sum(transposed_board[row][col:col + 4])

   # diagonal score \
   for row in range(len(state) - 3):
      for col in range(len(state[row]) - 3):
         for k in range(4):
            utility += state[row + k][col + k]

   # diagonal score /
   for row in range(len(state) - 3):
      for col in range(3, len(state[row])):
         for k in range(4):
            utility += state[row + k][col - k]

   return utility


def get_children(state, player):
   """ 
   returns a list of tuples containing children of state and what column the move in child was made in (child, col)
   """

   children = []

   for row in range(len(state)):
      for col in range(len(state[row])):
         if state[row][col] == 0 and row == 5:
            child = state.copy()
            child[row][col] = player
            children.append((child, col))

         elif state[row][col] == 0 and state[row+1][col] != 0:
            child = state.copy()
            child[row][col] = player
            children.append((child, col))

   return children

def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(state) # TODO: change input here

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
