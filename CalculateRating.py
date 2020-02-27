import matplotlib.pyplot as plt
import numpy as np
import trueskill as ts

import TerminatorHex
from HexBoard import HexBoard

"""This script calculates the rating of three Hex algorithms by playing them against each other and visualizes the 
evolution of their ratings.
"""


def play_1v1(player1_move, player1_rating, player2_move, player2_rating, cur_round):
    board_size = 4

    # Randomly select color
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
                     interactive_text=False, ai_move=None, blue_ai_move=blue_ai_move,
                     red_ai_move=red_ai_move, ai_color=None)

    winning_color = board.get_winning_color()
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
    terminator_depth_3 = TerminatorHex.TerminatorHex(3, True)
    terminator_depth_4 = TerminatorHex.TerminatorHex(4, True)

    random_player_move = terminator_depth_3.random_move
    dijkstra3_move = terminator_depth_3.terminator_move
    dijkstra4_move = terminator_depth_4.terminator_move

    random_player_rating = ts.Rating()
    dijkstra3_rating = ts.Rating()
    dijkstra4_rating = ts.Rating()

    random_player_desc = "Random AI"
    dijkstra3_desc = "Search depth 3 with Dijkstra evaluation"
    dijkstra4_desc = "Search depth 4 with Dijkstra evaluation"

    random_player_mu = [random_player_rating.mu]
    dijkstra3_mu = [dijkstra3_rating.mu]
    dijkstra4_mu = [dijkstra4_rating.mu]

    random_player_sigma = [random_player_rating.sigma]
    dijkstra3_sigma = [dijkstra3_rating.sigma]
    dijkstra4_sigma = [dijkstra4_rating.sigma]

    max_rounds = 5
    round_number = 0
    while round_number < max_rounds:
        print("Currently playing round number %d of %d" % (round_number + 1, max_rounds))

        # Random vs dijkstra3
        print("Playing", random_player_desc, "vs", dijkstra3_desc)
        random_player_rating, dijkstra3_rating = play_1v1(random_player_move, random_player_rating,
                                                          dijkstra3_move, dijkstra3_rating, round_number)

        # Dijkstra3 vs dijkstra4
        print("Playing", dijkstra3_desc, "vs", dijkstra4_desc)
        dijkstra3_rating, dijkstra4_rating = play_1v1(dijkstra3_move, dijkstra3_rating,
                                                      dijkstra4_move, dijkstra4_rating, round_number)

        # Random vs dijkstra4
        print("Playing", random_player_desc, "vs", dijkstra4_desc)
        random_player_rating, dijkstra4_rating = play_1v1(random_player_move, random_player_rating,
                                                          dijkstra4_move, dijkstra4_rating, round_number)

        random_player_mu.append(random_player_rating.mu)
        random_player_sigma.append(random_player_rating.sigma)
        dijkstra3_mu.append(dijkstra3_rating.mu)
        dijkstra3_sigma.append(dijkstra3_rating.sigma)
        dijkstra4_mu.append(dijkstra4_rating.mu)
        dijkstra4_sigma.append(dijkstra4_rating.sigma)

        round_number += 1

    print("Final ratings are:")
    print(random_player_desc, ": ", random_player_rating, sep="")
    print(dijkstra3_desc, ": ", dijkstra3_rating, sep="")
    print(dijkstra4_desc, ": ", dijkstra4_rating, sep="")

    # Plot rating evolution
    random_half_sigma = np.array(random_player_sigma) / 2
    dijkstra3_half_sigma = np.array(dijkstra3_sigma) / 2
    dijkstra4_half_sigma = np.array(dijkstra4_sigma) / 2

    plt.figure()
    plt.errorbar(range(max_rounds + 1), random_player_mu, yerr=random_half_sigma, label=random_player_desc, fmt='o')
    plt.errorbar(range(max_rounds + 1), dijkstra3_mu, yerr=dijkstra3_half_sigma, label=dijkstra3_desc, fmt='o')
    plt.errorbar(range(max_rounds + 1), dijkstra4_mu, yerr=dijkstra4_half_sigma, label=dijkstra4_desc, fmt='o')
    plt.xlabel("Round number")
    plt.ylabel("Rating")
    plt.ylim((0, 50))
    plt.legend()
    plt.show()
