import TerminatorHex
import MCTSHex
from HexBoard import *

import numpy as np
from functools import partial
import matplotlib.pyplot as plt

"""
This script evaluates positional advantage according to a heuristic evaluation function (see TerminatorHex)
for N_EPOCHS games with two AIs. This allows one to see strengths and weaknesses of AIs measured over the
span of games.
"""

# parameters
N_EPOCHS = 10
BOARD_SIZE = 6
AI_1 = MCTSHex.MCTSHex(500, 10)
AI_2 = MCTSHex.MCTSHex(5000, 10)
AI_1_MOVE = AI_1.MCTS_move
AI_2_MOVE = AI_2.MCTS_move
EVAL_MAX_CAP = BOARD_SIZE # the maximum value for the evaluation heuristic

# functs and stuff
eval_heuristic = TerminatorHex.dijkstra_score_heuristic # evaluation heuristic
max_turns = BOARD_SIZE**2 + 1 # just to be sure

def loop_hijack(board, AI_move_func, eval_heuristic, eval_values_vector):
    # 'hijacks' an AI move function to evaluate positional advantage first
    move = AI_move_func(board)
    turn = len(board.move_list) # determine number of moves made
    color = [HexBoard.RED, HexBoard.BLUE][board.blue_to_move]
    eval_val = eval_heuristic(board, color) # pre-move evaluation value
    if eval_val == float('inf'): # infinite value
        eval_val = EVAL_MAX_CAP # win
    elif eval_val == float('-inf'):
        eval_val = -EVAL_MAX_CAP # lose
    eval_values_vector[turn] = eval_val
    return move

# play games
eval_values_per_turn = np.zeros((N_EPOCHS, max_turns)) # eval values per turn per game
wins = [None for _ in range(N_EPOCHS)] # track wins per game
num_rounds_vec = [0 for _ in range(N_EPOCHS)] # how many rounds the games last
for t in range(N_EPOCHS):
    print("Round", t + 1)
    eval_vector = eval_values_per_turn[t, :] # mem pointer
    partial_1 = partial(loop_hijack, AI_move_func=AI_1_MOVE, eval_heuristic=eval_heuristic, eval_values_vector=eval_vector)
    partial_2 = partial(loop_hijack, AI_move_func=AI_2_MOVE, eval_heuristic=eval_heuristic, eval_values_vector=eval_vector)

    # do iteration
    board = HexBoard(BOARD_SIZE, n_players=0, enable_gui=False, interactive_text=True,
                 ai_move=partial_1, ai_color=HexBoard.BLUE,
                 blue_ai_move=partial_1, red_ai_move=partial_2,
                 move_list=[])
    
    record_win = board.get_winning_color()
    wins[t] = record_win
    num_rounds_vec[t] = len(board.move_list)

#print(eval_values_per_turn)
#print(wins)
wins_blue = wins.count(HexBoard.BLUE)
wins_red = wins.count(HexBoard.RED)
print("Wins per player:", "BLUE (P1):", wins_blue,
      "RED (P2):", wins_red)

# method of averaging
#average_advantages = np.average(eval_values_per_turn, axis=0) # average along time axis (epochs) # simple version
average_advantages = np.zeros((max_turns))
lasting_games = np.full((max_turns), N_EPOCHS) # number of games lasting until given turn
print("Game lengths:", num_rounds_vec)
for q in range(max_turns): # per turn
    for t in range(N_EPOCHS): # per game
        if num_rounds_vec[t] >= q: # if the game lasted until at least this turn
            average_advantages[q] += eval_values_per_turn[t, q] # increment by eval value
        else: # game was stopped at this turn already
            lasting_games[q] -= 1 # decrement number of games lasting until this turn by 1
for q in range(max_turns):
    if lasting_games[q] != 0: # div by zero
        average_advantages[q] = average_advantages[q] / lasting_games[q] # normalise
            
average_advantages_p1 = average_advantages[0::2] # for player 1 (once every two turns)
average_advantages_p2 = average_advantages[1::2]
fig, ax = plt.subplots()
ax.plot(average_advantages_p1, color='blue', label=("P1 (" + str(wins_blue) + " wins)"))
ax.plot(average_advantages_p2, color='red', label=("P2 (" + str(wins_red) + " wins)"))
ax.legend()
ax.set_title("Advantage plot")
ax.set_xlabel("Turn (per player!)")
ax.set_ylabel("Average heuristic advantage before move for setting player")
plt.show()
