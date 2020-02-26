import matplotlib.pyplot as plt
import numpy as np
import trueskill as ts

import TerminatorHex
from HexBoard import HexBoard

"""This script calculates the rating of two Hex algorithms by playing them against each other.
"""

if __name__ == '__main__':
    enable_GUI = False
    enable_interactive_text = True  # Todo: Needs to be true, but why?
    board_size = 4
    n_players = 0
    search_depth = 2

    terminator_AI = TerminatorHex.TerminatorHex(search_depth, True)
    terminator_depth_2 = TerminatorHex.TerminatorHex(2, True)
    terminator_depth_3 = TerminatorHex.TerminatorHex(3, True)

    player1_rating = ts.Rating()
    player2_rating = ts.Rating()
    player1_move = terminator_AI.terminator_move
    player2_move = terminator_depth_3.random_move
    player1_desc = "Terminator AI (depth = %d)" % search_depth
    player2_desc = "Random AI"

    player1_mu = [player1_rating.mu]
    player1_sigma = [player1_rating.sigma]
    player2_mu = [player2_rating.mu]
    player2_sigma = [player2_rating.sigma]

    max_games = 12
    game_number = 0
    while game_number < max_games:
        print("Currently playing game number %d of %d" % (game_number + 1, max_games))
        print("Current ratings are: ", player1_rating, " and ", player2_rating)

        # Randomly select color
        if game_number % 2 == 0:
            player1_color = HexBoard.BLUE
            player2_color = HexBoard.RED
            blue_ai_move = player1_move
            red_ai_move = player2_move
        else:
            player1_color = HexBoard.RED
            player2_color = HexBoard.BLUE
            blue_ai_move = player2_move
            red_ai_move = player1_move

        board = HexBoard(board_size, n_players=n_players, enable_GUI=enable_GUI,
                         interactive_text=enable_interactive_text, ai_move=None, blue_ai_move=blue_ai_move,
                         red_ai_move=red_ai_move, ai_color=None)

        winning_color = board.get_winning_color()
        if winning_color == player1_color:
            new_player1_rating, new_player2_rating = ts.rate_1vs1(player1_rating, player2_rating)
        elif winning_color == player2_color:
            new_player2_rating, new_player1_rating = ts.rate_1vs1(player2_rating, player1_rating)
        else:
            print("Rating error")

        player1_rating = new_player1_rating
        player2_rating = new_player2_rating

        player1_mu.append(player1_rating.mu)
        player1_sigma.append(player1_rating.sigma)
        player2_mu.append(player2_rating.mu)
        player2_sigma.append(player2_rating.sigma)

        game_number += 1

    print("Final ratings are:\n", player1_desc, ": ", player1_rating, "\n", player2_desc, ": ", player2_rating,
          sep="")

    player1_half_sigma = np.array(player1_sigma) / 2
    player2_half_sigma = np.array(player2_sigma) / 2
    plt.figure()
    plt.errorbar(range(max_games + 1), player1_mu, yerr=player1_half_sigma, label=player1_desc, fmt='o')
    plt.errorbar(range(max_games + 1), player2_mu, yerr=player2_half_sigma, label=player2_desc, fmt='o')
    plt.xlabel("Game number")
    plt.ylabel("Rating")
    plt.ylim((0, 50))
    plt.legend()
    plt.show()
