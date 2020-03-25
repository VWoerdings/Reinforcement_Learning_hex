import TerminatorHex
from HexBoard import HexBoard

"""
This script contains:
    the alpha-beta search function
    a move generator
    a random evaluator function
By executing this script you can play against an ai in a text-based interface. You can also enable a graphical
interface by setting enable_gui to True."""

if __name__ == '__main__':
    board_size = 4
    enable_gui = False

    if enable_gui:
        enable_text_interface = False
    else:
        enable_text_interface = True

    evaluator_function = TerminatorHex.random_score_heuristic  # Random evaluator function
    ai = TerminatorHex.TerminatorHex(3, use_suggested_heuristic=False, heuristic_evaluator=evaluator_function,
                                     depth_weighting=0, random_seed=10, do_iterative_deepening=False, max_time=None,
                                     do_transposition=False)
    move_generator = ai.terminator_move  # Move generator that uses alpha_beta search
    board = HexBoard(board_size, n_players=1, enable_gui=enable_gui, interactive_text=enable_text_interface,
                     ai_move=move_generator, ai_color=HexBoard.RED, blue_ai_move=None, red_ai_move=None, move_list=[])
