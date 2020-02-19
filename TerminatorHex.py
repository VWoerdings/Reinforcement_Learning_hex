import copy

class TerminatorHex:
    """This is a highly skilled hex game engine which will terminate opponent or terminate itself.
    """

    def __init__(self, boardData):
        """Initializes the Terminator AI for the HEX game
        Args:
            board_size (int): Size of the hexagon grid.
        """
        self.board = boardData

    def InitiateTerminator(self):
        """initiating distruction
        Args:
            board_size (int): Size of the hexagon grid.
        """
        return (2, 2)

    def terminateMinMax(self, depth):
        if (depth == 0 or self.board.gameOver):
            return
    
def minimax(hex_board, depth, max_or_min, evaluator): # 'HexBoard.BLUE', 'HexBoard.RED' for color; 'min' or 'max' to call
    """The minimax algorithm on the HexBoard
        Args:
            hex_board: start position
            depth: maximum depth to search
            max_or_min: maximise or minimise the current color to play (blue_to_move)
            evaluator: evaluator function. Called with args hex_board, maximiser_color
        Returns:
            maximised/minimised value according to the evaluator"""    
    moves = hex_board.get_free_positions()
    maximiser_color = [hex_board.BLUE, hex_board.RED][(max_or_min == 'max') ^ (hex_board.blue_to_move)]
    is_game_over = False
    if (hex_board.check_win(hex_board.BLUE) or hex_board.check_win(hex_board.RED)):
        is_game_over = True
    
    # minimax:
    if (depth <= 0 or is_game_over or len(moves) == 0): # end state
        return (evaluator(hex_board, maximiser_color))    
    elif (max_or_min == 'max'): # maximise
        value = float('-inf')
        for move in moves:
            deepened_board = copy.deepcopy(hex_board)
            deepened_board.set_position_auto(move)
            new_value = minimax(deepened_board, depth - 1, 'min', evaluator)
            value = [value, new_value][new_value > value]
        return value
    elif (max_or_min == 'min'): # minimise
        value = float('inf')
        for move in moves:
            deepened_board = copy.deepcopy(hex_board)
            deepened_board.set_position_auto(move)
            new_value = minimax(deepened_board, depth - 1, 'max', evaluator)
            value = [value, new_value][new_value < value]
        return value
    
    print("@minimax: unknown max_or_min objective", max_or_min)
    return None

def boardHash(hex_board, maximiser_color):
    """Hash the board, for deterministic random AI position evaluator"""
    board_list = []
    for k in hex_board.board.keys():
        board_list.append(hex_board.board[k])
    return hash(tuple(board_list))
