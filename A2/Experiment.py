from functools import partial

import MCTSHex
import TerminatorHex
import matplotlib.pyplot as plt
import numpy as np
import trueskill as ts
from HexBoard import HexBoard

"""
Find the optimal parameters for MCTS
"""


def get_elo_and_time(N, C_p, max_time=0, debug=False):
    if debug:
        print("Evaluting N=%d and C_p=%.3f" % (N, C_p))
    BOARD_SIZE = 6
    MAX_TURNS = BOARD_SIZE ** 2
    N_ROUNDS = 12

    terminator = TerminatorHex.TerminatorHex(2, do_transposition=False, max_time=max_time)
    terminator_player_move = terminator.terminator_move

    MCTS_AI = MCTSHex.MCTSHex(N, C_p, expansion_function=('constant', 1), random_seed="random", enh_WinScan=False,
                              enh_FreqVisitor=False, enh_EnsureTopLevelExplr=False)

    terminator_rating = ts.Rating()
    mcts_rating = ts.Rating()

    average_time = []
    for game in range(N_ROUNDS):
        if debug:
            print("Currently playing game number %d of %d" % (game + 1, N_ROUNDS))

        time_array = np.zeros((MAX_TURNS, 1))
        partial_move = partial(timeEvalMoveHook, AI_move_func=MCTS_AI.MCTS_move, timing_vector=time_array)

        if game % 2 == 0:
            mcts_color = HexBoard.BLUE
            terminator_color = HexBoard.RED
            blue_ai_move = partial_move
            red_ai_move = terminator_player_move
        else:
            mcts_color = HexBoard.RED
            terminator_color = HexBoard.BLUE
            blue_ai_move = terminator_player_move
            red_ai_move = partial_move

        board = HexBoard(BOARD_SIZE, n_players=0, enable_gui=False,
                         interactive_text=False, ai_color=None, ai_move=None, blue_ai_move=blue_ai_move,
                         red_ai_move=red_ai_move, move_list=[])
        winning_color = board.get_winning_color()

        if winning_color == mcts_color:
            mcts_rating, terminator_rating = ts.rate_1vs1(mcts_rating, terminator_rating)
        elif winning_color == terminator_color:
            terminator_rating, mcts_rating = ts.rate_1vs1(terminator_rating, mcts_rating)

        if mcts_color == HexBoard.BLUE:
            time_array = time_array[0::2]
        else:
            time_array = time_array[1::2]

        if debug:
            print("Average time was %.3f seconds" % np.mean(time_array))
        average_time.append(np.mean(time_array))
        mcts_trueskill = mcts_rating.mu - 3 * mcts_rating.sigma
        terminator_trueskill = terminator_rating.mu - 3 * terminator_rating.sigma
    return mcts_trueskill - terminator_trueskill, np.mean(average_time)


def play_1v1(player1_move, player1_rating, player2_move, player2_rating, cur_round):
    """Plays two AI algorithms against each other and updates their ratings.
    Args:
        player1_move (function): Move generator for player 1
        player1_rating (ts.Rating): Current rating for player 1
        player2_move (function): Move generator for player 2
        player2_rating (ts.Rating): Current rating for player 2
        cur_round (int): Current iteration number. Used to determine player colors.
    """
    board_size = 4

    # Select color
    if cur_round % 2 == 0:
        player1_color = HexBoard.BLUE
        player2_color = HexBoard.RED
        blue_ai_move = player1_move
        red_ai_move = player2_move
    else:
        player1_color = HexBoard.RED
        player2_color = HexBoard.BLUE
        blue_ai_move = player2_move
        red_ai_move = player1_move

    board = HexBoard(board_size, n_players=0, enable_gui=False,
                     interactive_text=False, ai_color=None, ai_move=None, blue_ai_move=blue_ai_move,
                     red_ai_move=red_ai_move, move_list=[])
    winning_color = board.get_winning_color()

    # Update ratings
    if winning_color == player1_color:
        new_player1_rating, new_player2_rating = ts.rate_1vs1(player1_rating, player2_rating)
    elif winning_color == player2_color:
        new_player2_rating, new_player1_rating = ts.rate_1vs1(player2_rating, player1_rating)
    else:
        new_player1_rating = None
        new_player2_rating = None
        print("Rating error")

    return new_player1_rating, new_player2_rating


if __name__ == '__main__':
    idtt_depths = [2, 3]
    mcts_settings = [(2158, 0.396), (2084, 0.942)]  # , (1387, 2.962)]

    MAX_GAMES = 25
    BOARD_SIZE = 6

    results = np.asarray([list(range(MAX_GAMES))])
    results_backup = np.ones((MAX_GAMES, 4))
    results_backup[:, 0] = results
    numm = 1
    for depth in idtt_depths:
        for N, C_p in mcts_settings:
            print("Currently playing search depth %d vs N=%d, C_p=%.3f" % (depth, N, C_p))
            rating_list = []
            terminator_rating = ts.Rating()
            mcts_rating = ts.Rating()
            for game in range(MAX_GAMES):
                print("Currently plaing game number %d of %d." % (game, MAX_GAMES))
                terminator = TerminatorHex.TerminatorHex(depth, do_transposition=True, do_iterative_deepening=True,
                                                         max_time=2)
                terminator_player_move = terminator.terminator_move

                MCTS_AI = MCTSHex.MCTSHex(N, C_p, expansion_function=('constant', 1), random_seed="random",
                                          enh_WinScan=False, enh_FreqVisitor=False, enh_EnsureTopLevelExplr=False)

                if game % 2 == 0:
                    mcts_color = HexBoard.BLUE
                    terminator_color = HexBoard.RED
                    blue_ai_move = MCTS_AI.MCTS_move
                    red_ai_move = terminator_player_move
                else:
                    mcts_color = HexBoard.RED
                    terminator_color = HexBoard.BLUE
                    blue_ai_move = terminator_player_move
                    red_ai_move = MCTS_AI.MCTS_move

                board = HexBoard(BOARD_SIZE, n_players=0, enable_gui=False,
                                 interactive_text=False, ai_color=None, ai_move=None, blue_ai_move=blue_ai_move,
                                 red_ai_move=red_ai_move, move_list=[])
                winning_color = board.get_winning_color()

                if winning_color == mcts_color:
                    mcts_rating, terminator_rating = ts.rate_1vs1(mcts_rating, terminator_rating)
                elif winning_color == terminator_color:
                    terminator_rating, mcts_rating = ts.rate_1vs1(terminator_rating, mcts_rating)

                rating_list.append(terminator_rating.mu)

            results = np.append(results, np.asarray([rating_list]), axis=0)
            # results_backup[:, numm] = np.asarray([rating_list])
            # numm += 1
            print(rating_list)

    idtt_depths = [2, 3]
    mcts_settings = [(2158, 0.396), (2084, 0.942)]  # , (1387, 2.962)]

    plt.figure()
    plt.plot(results[:, 0], results[:, 1],
             label="N=%d, C_p=%.3f vs search depth %d" % (2158, 0.396, 2))
    plt.plot(results[:, 0], results[:, 2],
             label="N=%d, C_p=%.3f vs search depth %d" % (2158, 0.396, 3))
    plt.plot(results[:, 0], results[:, 3],
             label="N=%d, C_p=%.3f vs search depth %d" % (2084, 0.942, 2))
    plt.plot(results[:, 0], results[:, 4],
             label="N=%d, C_p=%.3f vs search depth %d" % (2084, 0.942, 3))
    plt.legend()
    plt.title("Skill rating of MCTS vs number of rounds played")
    plt.xlabel("Game number")
    plt.ylabel("Rating")
    plt.show()
