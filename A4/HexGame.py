import numpy as np

import TerminatorHex
import MCTSHex
from Hex import *

if __name__ == '__main__':
    enable_GUI = True
    enable_interactive_text = True
    board_size = 5
    n_players = 0
    ai_color = HexBoard.RED
    terminator_AI = TerminatorHex.TerminatorHex(3, True, random_seed='random', do_transposition=True, do_iterative_deepening=True)
    mcts_AI = MCTSHex.MCTSHex(500, 10, expansion_function=('constant', 1), enh_WinScan=True, enh_EnsureTopLevelExplr=True)
    board = HexUI(board_size, n_players=n_players, enable_gui=enable_GUI, interactive_text=enable_interactive_text,
                     ai_move=terminator_AI.terminator_move, ai_color=ai_color,
                     blue_ai_move=terminator_AI.terminator_move, red_ai_move=mcts_AI.MCTS_move,
                     move_list=[])

    if not enable_GUI and not enable_interactive_text:

        # sanity check that wins are detected
        for i in range(0, 2):
            winner = HexBoard.RED if i == 0 else HexBoard.BLUE
            loser = HexBoard.BLUE if i == 0 else HexBoard.RED
            board = HexBoard(3)
            board.place_with_color((1, 1), loser)
            board.place_with_color((2, 1), loser)
            board.place_with_color((1, 2), loser)
            board.place_with_color((2, 2), loser)
            board.place_with_color((0, 0), winner)
            board.place_with_color((1, 0), winner)
            board.place_with_color((2, 0), winner)
            board.place_with_color((0, 1), winner)
            board.place_with_color((0, 2), winner)
            assert (board.check_win(winner) == True)
            assert (board.check_win(loser) == False)
            board.print_board()
        endable_board = HexBoard(4)
        # sanity check that random play will at some point end the game
        while not endable_board.game_over:
            endable_board.place_with_color((np.random.randint(0, 4), np.random.randint(0, 4)), HexBoard.RED)
        assert (endable_board.game_over == True)
        assert (endable_board.check_win(HexBoard.RED) == True)
        assert (endable_board.check_win(HexBoard.BLUE) == False)
        print("Randomly filled board")
        endable_board.print_board()

        neighbor_check = HexBoard(5)
        assert (neighbor_check.get_neighbors((0, 0)) == [(1, 0), (0, 1)])
        assert (neighbor_check.get_neighbors((0, 1)) == [(1, 1), (1, 0), (0, 2), (0, 0)])
        assert (neighbor_check.get_neighbors((1, 1)) == [(0, 1), (2, 1), (0, 2), (2, 0), (1, 2), (1, 0)])
        assert (neighbor_check.get_neighbors((3, 4)) == [(2, 4), (4, 4), (4, 3), (3, 3)])
        assert (neighbor_check.get_neighbors((4, 3)) == [(3, 3), (3, 4), (4, 4), (4, 2)])
        assert (neighbor_check.get_neighbors((4, 4)) == [(3, 4), (4, 3)])
        neighbor_check_11 = HexBoard(5)
        assert (neighbor_check_11.get_neighbors((4, 4)) == [(3, 4), (4, 3)])

        neighbor_check_small = HexBoard(2)
        assert (neighbor_check_small.get_neighbors((0, 0)) == [(1, 0), (0, 1)])
        assert (neighbor_check_small.get_neighbors((1, 0)) == [(0, 0), (0, 1), (1, 1)])
        assert (neighbor_check_small.get_neighbors((0, 1)) == [(1, 1), (1, 0), (0, 0)])
        assert (neighbor_check_small.get_neighbors((1, 1)) == [(0, 1), (1, 0)])

        neighbor_check_sanity = HexBoard(11)
        for x in range(0, 11):
            for y in range(0, 11):
                neighbors = neighbor_check_sanity.get_neighbors((x, y))
                for neighbor in neighbors:
                    neighbors_neighbors = neighbor_check_sanity.get_neighbors(neighbor)
                    index_of_self = neighbors_neighbors.index((x, y))
                    assert (index_of_self != -1)

# main() # replaced by __name__ == '__main__'
