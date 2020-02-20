import copy

class TerminatorHex:
    """This is a highly skilled hex game engine which will terminate opponent or terminate itself.
    """

    def __init__(self, max_depth, heuristic_evaluator):
        """Initializes the Terminator AI for the HEX game
        Args:
            board: the HexBoard class to which we bind
            max_depth: max search depth
            heuristic_evaluator: heuristic evaluation function (args hex_board, maximiser_color)
        """
        #self.board = board
        self.max_depth = max_depth
        self.heuristic_evaluator = heuristic_evaluator

    def terminator_move(self, board):
        """initiating destruction
        Return the best move according to TerminatorHex on self.board for the current color to move (blue_to_move on the board)
        """
        return self.terminator_min_max(board, self.max_depth, 'max')

    def terminator_min_max(self, hex_board, depth, max_or_min):
        alpha = float('-inf') # initial alpha beta bounds
        beta = float('inf')
        moves = hex_board.get_free_positions()
        
        is_game_over = False
        if (hex_board.check_win(hex_board.BLUE) or hex_board.check_win(hex_board.RED)):
            is_game_over = True
            
        if (depth == 0 or is_game_over or len(moves) == 0): # leaf node or game over
            print("@TerminatorHex.terminator_min_max: no valid moves left, or out of depth")
            return None # shouldn't happen
        elif (max_or_min == 'max'): # maximise
            value = float('-inf')
            best_move = None
            for move in moves:
                deepened_board = copy.deepcopy(hex_board)
                deepened_board.set_position_auto(move)
                #new_value = minimax(deepened_board, depth - 1, 'min', self.heuristic_evaluator)
                new_value = alpha_beta(deepened_board, depth - 1, 'min', alpha, beta, self.heuristic_evaluator)
                if (new_value > value):
                    value = new_value
                    best_move = move
            return best_move
        elif (max_or_min == 'min'): # minimise
            value = float('inf')
            best_move = None
            for move in moves:
                deepened_board = copy.deepcopy(hex_board)
                deepened_board.set_position_auto(move)
                #new_value = minimax(deepened_board, depth - 1, 'max', self.heuristic_evaluator)
                new_value = alpha_beta(deepened_board, depth - 1, 'max', alpha, beta, self.heuristic_evaluator)
                if (new_value < value):
                    value = new_value
                    best_move = move
            return best_move
    
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

def alpha_beta(hex_board, depth, max_or_min, alpha, beta, evaluator):
    """The minimax algorithm on the HexBoard with alpha-beta-pruning
    Args:
        hex_board: start position
        depth: maximum depth to search
        max_or_min: maximise or minimise the current color to play (blue_to_move)
        alpha, beta: initial alpha and beta bounds. use float('-inf') and float('inf')
        evaluator: evaluator function. Called with args hex_board, maximiser_color
    Returns:
        maximised/minimised value according to the evaluator"""
    moves = hex_board.get_free_positions()
    maximiser_color = [hex_board.BLUE, hex_board.RED][(max_or_min == 'max') ^ (hex_board.blue_to_move)]
    is_game_over = False
    if (hex_board.check_win(hex_board.BLUE) or hex_board.check_win(hex_board.RED)):
        is_game_over = True
    
    # minimax with alpha-beta pruning:
    if (depth <= 0 or is_game_over or len(moves) == 0): # end state
        return (evaluator(hex_board, maximiser_color))    
    elif (max_or_min == 'max'): # maximise
        value = float('-inf')
        for move in moves:
            deepened_board = copy.deepcopy(hex_board)
            deepened_board.set_position_auto(move)
            new_value = minimax(deepened_board, depth - 1, 'min', evaluator)
            value = [value, new_value][new_value > value]
            alpha = [alpha, new_value][new_value > alpha]
            if (alpha >= beta): # beta cutoff
                break
        return value
    elif (max_or_min == 'min'): # minimise
        value = float('inf')
        for move in moves:
            deepened_board = copy.deepcopy(hex_board)
            deepened_board.set_position_auto(move)
            new_value = minimax(deepened_board, depth - 1, 'max', evaluator)
            value = [value, new_value][new_value < value]
            beta = [beta, new_value][new_value < beta]
            if (alpha >= beta): # alpha cutoff
                break
        return value
    
    print("@alpha_beta: unknown max_or_min objective", max_or_min)
    return None

def board_hash(hex_board, maximiser_color):
    """Hash the board, for deterministic random AI position evaluator"""
    board_list = []
    for k in hex_board.board.keys():
        board_list.append(hex_board.board[k])
    return hash(tuple(board_list))

#def dijkstra_score_heuristic(hex_board, maximiser_color)
#def board_dijkstra(hex_board, maximiser_color)
