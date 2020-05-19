import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import trueskill as ts

sys.path.append('..')

from hex.HexGame import HexGame
from hex.HexBoard import HexBoard
from hex.NNet import NNetWrapper
from MCTS import MCTS
from utils import *

from old_ai.HexBoard_old import HexBoard as HB
from old_ai.TerminatorHex import TerminatorHex
from old_ai.MCTSHex import MCTSHex

"""
Find the optimal parameters for MCTS
"""


def alphazero_move(mcts, board_size):
    def move_gen(hexboard):
        # Generate a move using alphazero ai
        dict_board = hexboard.board
        array_board = np.zeros((board_size, board_size))
        for key in dict_board.keys():
            # Convert the board from dict to array
            x, y = key
            array_board[x, y] = dict_board[x, y]
        action = np.argmax(mcts.getActionProb(array_board, temp=0))
        move = (int(action / board_size), action % board_size)  # Convert the action from int to tuple
        return move

    return move_gen


if __name__ == '__main__':

    BOARD_SIZE = 7
    MAX_GAMES = 25  # Number of games to play
    print_time = True

    game = HexGame(BOARD_SIZE)
    args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})

    # Alphazero player
    n1 = NNetWrapper(game)
    n1.load_checkpoint('./trained_networks/', 'best_100iters_run1.pth.tar')
    mcts1 = MCTS(game, n1, args)
    n1p = alphazero_move(mcts1, BOARD_SIZE)  # This is the move generating function
    alphazero_description1 = "Alphazero player 100 iterations (run 1)"

    # Alphazero player
    n2 = NNetWrapper(game)
    n2.load_checkpoint('./trained_networks/', 'best_100iters_run2.pth.tar')
    mcts2 = MCTS(game, n2, args)
    n2p = alphazero_move(mcts2, BOARD_SIZE)  # This is the move generating function
    alphazero_description2 = "Alphazero player 100 iterations (run 2)"

    # ID-TT player
    depth = 2
    idtt = TerminatorHex(depth, do_transposition=True, do_iterative_deepening=True, max_time=2)
    idtt_move = idtt.terminator_move  # This is the move generating function
    idtt_description = "ID-TT player"

    # MCTS player
    N = 2158
    C_p = 0.396
    MCTS_AI = MCTSHex(N, C_p, expansion_function=('constant', 1), random_seed="random",
                      enh_WinScan=False, enh_FreqVisitor=False, enh_EnsureTopLevelExplr=False)
    mcts_move = MCTS_AI.MCTS_move  # This is the move generating function
    mcts_description = "MCTS player"

    # Define players here
    player1_move = n1p  # Replace with desired move generating function
    player2_move = n2p  # Replace with desired move generating function

    # Descriptions for plot legend
    player1_description = alphazero_description1
    player2_description = alphazero_description2

    # Lists storing the results of the ratings
    p1_rating_list = []
    p2_rating_list = []
    p1_rating = ts.Rating()
    p2_rating = ts.Rating()

    for game_n in range(MAX_GAMES):
        print("Currently playing game number %d of %d..." % (game_n, MAX_GAMES))
        start = time.time()

        if game_n % 2 == 0:
            # arena = Arena(n1p, n2p, game)
            p1_color = HexBoard.BLUE
            p2_color = HexBoard.RED
            blue_ai_move = player1_move
            red_ai_move = player2_move
        else:
            # arena = Arena(n2p, n1p, game)
            p1_color = HexBoard.RED
            p2_color = HexBoard.BLUE
            blue_ai_move = player2_move
            red_ai_move = player1_move

        board = HB(BOARD_SIZE, n_players=0, enable_gui=False,
                   interactive_text=False, ai_color=None, ai_move=None, blue_ai_move=blue_ai_move,
                   red_ai_move=red_ai_move, move_list=[])
        winning_color = board.get_winning_color()

        if winning_color == p1_color:
            p1_rating, p2_rating = ts.rate_1vs1(p1_rating, p2_rating)
        elif winning_color == p2_color:
            p2_rating, p1_rating = ts.rate_1vs1(p2_rating, p1_rating)
        else:
            raise ValueError("There was a draw or an unexpected winner: %d" % winning_color)

        end = time.time()
        if print_time:
            print("Time elapsed is %.2f s" % (end - start))

        p1_rating_list.append(p1_rating.mu)
        p2_rating_list.append(p2_rating.mu)

    plt.figure()
    plt.plot(np.squeeze(np.asarray([range(MAX_GAMES)])), np.squeeze(np.asarray(p1_rating_list)),
             label=player1_description)
    plt.plot(np.squeeze(np.asarray([range(MAX_GAMES)])), np.squeeze(np.asarray(p2_rating_list)),
             label=player2_description)
    plt.legend()
    plt.title("Skill rating of 2 AI algorithms vs number of rounds played")
    plt.xlabel("Game number")
    plt.ylabel("Rating")
    plt.show()
