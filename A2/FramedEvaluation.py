import TerminatorHex
import MCTSHex
from HexBoard import *

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import os

"""
This script evaluates positional advantage according to a heuristic evaluation function (see TerminatorHex)
for N_EPOCHS games with two AIs. This allows one to see strengths and weaknesses of AIs measured over the
span of games.
"""

# parameters
N_EPOCHS = 2
BOARD_SIZE = 6
AI_1 = MCTSHex.MCTSHex(500, 10, expansion_function=('constant', 0.7), enh_FreqVisitor=False, enh_WinScan=True)
AI_2 = MCTSHex.MCTSHex(500, 10, expansion_function=('constant', 1))
AI_1_MOVE = AI_1.MCTS_move
AI_2_MOVE = AI_2.MCTS_move
EVAL_MAX_CAP = BOARD_SIZE # the maximum value for the evaluation heuristic
PLOT_NUM_GAMES = True # plot the number of games on a second y-axis
TRACK_TREE_SIZE = True # TODO: track tree size in the loop_hijack function, plot it separately
SAVE_PLOTS = True # save the plots to a folder
SAVE_DIR = os.getcwd() + "/graphs"

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# functs and stuff
eval_heuristic = TerminatorHex.dijkstra_score_heuristic # evaluation heuristic
max_turns = BOARD_SIZE**2 + 1 # just to be sure

def loop_hijack(board, AI_move_func, eval_heuristic, eval_values_vector, AI=None, tree_size_vec=None):
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

    if TRACK_TREE_SIZE:
        num_tree_nodes = sum(MCTSHex.get_MCTSNode_level_table(AI.tree_head)) # sum the total number of nodes in the tree
        tree_size_vec[turn] = num_tree_nodes
        
    return move

# play games
eval_values_per_turn = np.zeros((N_EPOCHS, max_turns)) # eval values per turn per game
tree_size_per_turn = np.zeros((N_EPOCHS, max_turns)) # may be unused if TRACK_TREE_SIZE is not True
wins = [None for _ in range(N_EPOCHS)] # track wins per game
num_rounds_vec = [0 for _ in range(N_EPOCHS)] # how many rounds the games last
for t in range(N_EPOCHS):
    print("Round", t + 1)
    eval_vector = eval_values_per_turn[t, :] # mem pointer
    tree_vector = tree_size_per_turn[t, :] # mem pointer
    partial_1 = partial(loop_hijack, AI_move_func=AI_1_MOVE, eval_heuristic=eval_heuristic, eval_values_vector=eval_vector, AI=AI_1, tree_size_vec=tree_vector)
    partial_2 = partial(loop_hijack, AI_move_func=AI_2_MOVE, eval_heuristic=eval_heuristic, eval_values_vector=eval_vector, AI=AI_2, tree_size_vec=tree_vector)

    # do iteration
    board = HexBoard(BOARD_SIZE, n_players=0, enable_gui=False, interactive_text=True,
                 ai_move=partial_1, ai_color=HexBoard.BLUE,
                 blue_ai_move=partial_1, red_ai_move=partial_2,
                 move_list=[])
    
    record_win = board.get_winning_color()
    wins[t] = record_win
    num_rounds_vec[t] = len(board.move_list)

    if type(AI_1) == MCTSHex.MCTSHex:
        AI_1.cull_tree() # destroy the current tree
    if type(AI_2) == MCTSHex.MCTSHex:
        AI_2.cull_tree()

#print(eval_values_per_turn)
#print(wins)
wins_blue = wins.count(HexBoard.BLUE)
wins_red = wins.count(HexBoard.RED)
print("Wins per player:", "BLUE (P1):", wins_blue,
      "RED (P2):", wins_red)

# method of averaging
#average_advantages = np.average(eval_values_per_turn, axis=0) # average along time axis (epochs) # simple version
average_advantages = np.zeros((max_turns))
average_tree_sizes = np.zeros((max_turns)) # average the tree sizes, if TRACK_TREE_SIZE is used
lasting_games = np.full((max_turns), N_EPOCHS) # number of games lasting until given turn
print("Game lengths:", num_rounds_vec)
for q in range(max_turns): # per turn
    for t in range(N_EPOCHS): # per game
        if num_rounds_vec[t] >= q: # if the game lasted until at least this turn
            average_advantages[q] += eval_values_per_turn[t, q] # increment by eval value
            if TRACK_TREE_SIZE:
                average_tree_sizes[q] += tree_size_per_turn[t, q]
        else: # game was stopped at this turn already
            lasting_games[q] -= 1 # decrement number of games lasting until this turn by 1
for q in range(max_turns):
    if lasting_games[q] != 0: # div by zero
        average_advantages[q] = average_advantages[q] / lasting_games[q] # normalise
        if TRACK_TREE_SIZE:
            average_tree_sizes[q] = average_tree_sizes[q] / lasting_games[q]
            
average_advantages_p1 = average_advantages[0::2] # for player 1 (once every two turns)
average_advantages_p2 = average_advantages[1::2] # player 2
if TRACK_TREE_SIZE:
    average_tree_sizes_p1 = average_tree_sizes[0::2] # for player 1 (once every two turns)
    average_tree_sizes_p2 = average_tree_sizes[1::2] # player 2
    
fig, ax = plt.subplots()
ax.plot(average_advantages_p1, color='blue', label=("P1 (" + str(wins_blue) + " wins)"))
ax.plot(average_advantages_p2, color='red', label=("P2 (" + str(wins_red) + " wins)"))
ax.plot([0], 'g--', label="Lasting games") # ghost plot
ax.legend()
ax.set_title("Advantage plot")
ax.set_xlabel("Turn (per player!)")
ax.set_ylabel("Average heuristic advantage before move for setting player")

if PLOT_NUM_GAMES: # plot games on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(lasting_games[0::2], 'g--', label="Lasting games") # from blue player perspective
    ax2.set_ylabel("Number of remaining games at this turn (blue player)")
    #ax2.legend()

if TRACK_TREE_SIZE:
    fig2, ax_f2 = plt.subplots()
    ax_f2.plot(average_tree_sizes_p1, color='blue', label=("P1 (" + str(wins_blue) + " wins)"))
    ax_f2.plot(average_tree_sizes_p2, color='red', label=("P2 (" + str(wins_red) + " wins)"))
    ax_f2.plot([0], 'g--', label="Lasting games") # ghost plot
    ax_f2.legend()
    ax_f2.set_title("Average tree size per turn")
    ax_f2.set_xlabel("Turn (per player!)")
    ax_f2.set_ylabel("Average tree size (every level)")

    if PLOT_NUM_GAMES: # plot games on secondary y-axis
        ax2_f2 = ax_f2.twinx()
        ax2_f2.plot(lasting_games[0::2], 'g--', label="Lasting games") # from blue player perspective
        ax2_f2.set_ylabel("Number of remaining games at this turn (blue player)")
        #ax2_f2.legend()

if SAVE_PLOTS:
    # systematic name
    def getSystematicID(AI):
        explr_fnc_infix = ["s", "l"][AI.expansion_function_mode == 'lambda']
        explr_fnc_infix = [explr_fnc_infix, "f"][AI.expansion_function_mode == 'constant']
        if explr_fnc_infix in ["s", "f"]:
            explr_fnc_infix += str(int(1 / AI.expansion_function_const))
        if AI.enh_WinScan:
            explr_fnc_infix += "w"
        if AI.enh_FreqVisitor:
            explr_fnc_infix += "v"
        if AI.enh_EnsureTopLevelExplr:
            explr_fnc_infix += "e"
        explr_fnc_infix += ("c" + str(AI.c_explore))
        return explr_fnc_infix

    # saving system
    sysname_1 = getSystematicID(AI_1)
    sysname_2 = getSystematicID(AI_2)
    
    name_infix = str(N_EPOCHS) + "_" + str(AI_1.N_trials) + sysname_1 + "_" + str(AI_2.N_trials) + sysname_2
    name_A = "A" + name_infix + ".png"
    if TRACK_TREE_SIZE:
        name_T = "T" + name_infix + ".png"
        
    postfix = 1
    while True: # attempt to save
        if not os.path.isfile(SAVE_DIR + "/" + name_A):
            if TRACK_TREE_SIZE:
                if not os.path.isfile(SAVE_DIR + "/" + name_T):
                    fig.savefig((SAVE_DIR + "/" + name_A))
                    fig2.savefig((SAVE_DIR + "/" + name_T))
                    break
            else:
                fig.savefig((SAVE_DIR + "/" + name_A))
                break

        name_A = ("A" + name_infix + "_(" + str(postfix) + ")" + ".png")
        if TRACK_TREE_SIZE:
            name_T = ("T" + name_infix + "_(" + str(postfix) + ")" + ".png")
        postfix += 1
        
        if postfix > 100000: # safeguard
            print("@FramedEvaluation>SAVE_PLOTS section: failed to save")
            break

plt.show()
