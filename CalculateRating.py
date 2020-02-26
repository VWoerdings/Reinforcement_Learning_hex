import numpy as np
import trueskill as ts

import TerminatorHex
from HexBoard import HexBoard


if __name__ == '__main__':
    enable_GUI = False
    enable_interactive_text = True  # Needs to be true, but why?
    board_size = 4
    n_players = 0
    search_depth = 2

    terminator_AI = TerminatorHex.TerminatorHex(search_depth, True)

    player1_rating = ts.Rating()
    player1_move = terminator_AI.random_move
    player2_rating = ts.Rating()
    player2_move = terminator_AI.terminator_move

    max_games = 12
    game_number = 0
    while game_number < max_games:
        print("Currently playing game number %d of %d" % (game_number+1, max_games))
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

        game_number += 1

    print("Final ratings are: \nRandom player:", player1_rating, " and Terminator with search depth 2:", player2_rating)
