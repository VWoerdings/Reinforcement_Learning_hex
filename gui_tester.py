import numpy as np

import TerminatorHex
from HexBoard import HexBoard

""" Simple class to play a single player game against an AI using the GUI.
"""

if __name__ == '__main__':
    enable_GUI = True
    enable_interactive_text = True
    board_size = 5
    n_players = 1
    ai_color = HexBoard.BLUE

    terminator_AI = TerminatorHex.TerminatorHex(3, True)
    board = HexBoard(board_size, n_players=n_players, enable_gui=enable_GUI, interactive_text=enable_interactive_text,
                     ai_move=terminator_AI.terminator_move, ai_color=ai_color, blue_ai_move=None, red_ai_move=None,
                     move_list=[])
