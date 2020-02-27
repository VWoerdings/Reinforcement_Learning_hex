import numpy as np

import TerminatorHex
from HexBoard import HexBoard


if __name__ == '__main__':
    enable_GUI = True
    enable_interactive_text = False
    board_size = 5
    n_players = 1
    ai_color = HexBoard.BLUE

    terminator_AI = TerminatorHex.TerminatorHex(2, True)
    board = HexBoard(board_size, n_players=n_players, enable_gui=enable_GUI, interactive_text=enable_interactive_text,
                     ai_move=terminator_AI.terminator_move, ai_color=ai_color)
