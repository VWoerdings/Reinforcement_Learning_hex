import time
from functools import partial

import MCTSHex
import TerminatorHex
import numpy as np
import trueskill as ts
from HexBoard import HexBoard

"""
Find the optimal parameters for MCTS
"""


def get_score(N, C_p, max_time=0, debug=False):
    """
    Score for the current parameters
    :return:
    """
    penalty = 10 ** -3
    ini = time.time()
    elo, runtime = get_elo_and_time(N, C_p, max_time=max_time, debug=debug)
    fin = time.time()
    if debug:
        print("One eval took %.1f minutes" % ((fin - ini)/60))
    if max_time != 0:
        if runtime > max_time:
            return penalty * elo / runtime
    return elo / runtime


def timeEvalMoveHook(board, AI_move_func, timing_vector):
    """
    This function 'hijacks' a normal AI move function to evaluate the time taken for computation of that move.
    Via functools.partial: Supply an AI move function (instantiate an AI class first) and a timing vector (must be passed by reference,
    e.g. as numpy array!) which stores the evaluation times per turn.
    Then the partial can be provided to the HexBoard class as a move function with only board as argument.
    Args:
        board (HexBoard): the current HexBoard. Can be provided by the interactive loop of the HexBoard class.
        AI_move_func (Callable, must return HexBoard move): a Hex AI instanced move function. Partialise.
        timing_vector (array): an array to store timing values per turn, pass by reference, i.e. as numpy array.
    Returns:
        (int, int): a valid HexBoard move.
    """

    start = time.time()
    move = AI_move_func(board)
    end = time.time()
    turn = len(board.move_list)  # determine number of moves made
    timing_vector[turn] = (end - start)  # store time taken
    return move  # return the 'hijacked' move


def get_elo_and_time(N, C_p, max_time=0, debug=False):
    if debug:
        print("Evaluting N=%d and C_p=%.3f" % (N, C_p))
    BOARD_SIZE = 6
    MAX_TURNS = BOARD_SIZE ** 2
    N_ROUNDS = 12

    terminator = TerminatorHex.TerminatorHex(2, do_transposition=False, max_time=max_time)
    random_player_move = terminator.terminator_move

    MCTS_AI = MCTSHex.MCTSHex(N, C_p, expansion_function=('constant', 1), random_seed="random", enh_WinScan=False,
                              enh_FreqVisitor=False, enh_EnsureTopLevelExplr=False)

    random_rating = ts.Rating()
    mcts_rating = ts.Rating()

    average_time = []
    for game in range(N_ROUNDS):
        if debug:
            print("Currently playing game number %d of %d" % (game + 1, N_ROUNDS))

        time_array = np.zeros((MAX_TURNS, 1))
        partial_move = partial(timeEvalMoveHook, AI_move_func=MCTS_AI.MCTS_move, timing_vector=time_array)

        if game % 2 == 0:
            mcts_color = HexBoard.BLUE
            random_color = HexBoard.RED
            blue_ai_move = partial_move
            red_ai_move = random_player_move
        else:
            mcts_color = HexBoard.RED
            random_color = HexBoard.BLUE
            blue_ai_move = random_player_move
            red_ai_move = partial_move

        board = HexBoard(BOARD_SIZE, n_players=0, enable_gui=False,
                         interactive_text=False, ai_color=None, ai_move=None, blue_ai_move=blue_ai_move,
                         red_ai_move=red_ai_move, move_list=[])
        winning_color = board.get_winning_color()

        if winning_color == mcts_color:
            mcts_rating, random_rating = ts.rate_1vs1(mcts_rating, random_rating)
        elif winning_color == random_color:
            random_rating, mcts_rating = ts.rate_1vs1(random_rating, mcts_rating)

        if debug:
            print("Average time was %.3f seconds" % np.mean(time_array[time_array != 0]))

        if mcts_color == HexBoard.BLUE:
            time_array = time_array[0::2]
        else:
            time_array = time_array[1::2]
        average_time.append(np.mean(time_array))
    return mcts_rating.mu - random_rating.mu, np.mean(average_time)


def insert_score(history, N, C_p, score, debug=False, check_top=False, K=0):
    if len(history) < 1:
        history = np.asarray([N, C_p, score]).reshape((1, 3))
    else:
        sorted_index = np.searchsorted(history[:, 2], score)
        prev = history[:sorted_index, :]
        mid = np.asarray([N, C_p, score]).reshape((1, 3))
        post = history[sorted_index:, :]
        if debug:
            print("prev shape %s" % (prev.shape,))
            print("mid shape %s" % (mid.shape,))
            print("post shape %s" % (post.shape,))
        if sorted_index == 0:
            history = np.concatenate((mid, post), axis=0)
        elif sorted_index == len(history):
            history = np.concatenate((prev, mid), axis=0)
        else:
            history = np.concatenate((prev, mid, post), axis=0)

        if check_top:
            return history, (len(history) - sorted_index <= K)
    if debug:
        print(history)
    return history


if __name__ == '__main__':
    N_min = 100
    N_max = 1000
    C_p_min = 0
    C_p_max = 5

    top_K = 5
    initial_tries = 3
    MAX_TIME = 2

    debug = True

    history = []
    for C_p in np.linspace(C_p_min, C_p_max, num=initial_tries):
        for N in np.rint(np.linspace(N_min, N_max, num=initial_tries)).astype(np.int64):
            score = get_score(N, C_p, max_time=MAX_TIME, debug=debug)
            history = insert_score(history, N, C_p, score, debug)

    is_top = True
    C_p_convergence = 0.1
    iterations = 0
    max_iterations = 100
    while is_top and (iterations < max_iterations):
        top_N = history[-top_K:, 0]
        top_C_p = history[-top_K:, 1]
        top_scores = history[-top_K:, 2]

        new_N = int(np.dot(top_N, top_scores) / np.sum(top_scores))
        new_C_p = int(np.dot(top_C_p, top_scores) / np.sum(top_scores))

        score = get_score(new_N, new_C_p, max_time=MAX_TIME, debug=debug)
        history, is_top = insert_score(history, new_N, new_C_p, score, debug, True, top_K)
        print(history)
        iterations += 1
    if debug:
        print("Done after %d iterations" % iterations)

    print(history)
