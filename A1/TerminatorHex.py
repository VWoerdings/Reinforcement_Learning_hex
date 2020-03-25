import random
import time
from HexBoard import HexBoard


class TerminatorHex:
    """A class that computes the the best move to make.
    Contains the following evaluator functions:
        dijkstra_score_heuristic: Difference in path length between current player and opponent.
        Todo: update
    Contains the following move generators:
        random_move: Returns a non-deterministic random move.
        terminator_move: Uses the alpha-beta algorithm and the chosen heuristic evaluator to compute the best move.
    """

    def __init__(self, max_depth, use_suggested_heuristic=True, heuristic_evaluator=None, depth_weighting=0,
                 random_seed=100,
                 do_iterative_deepening=True, max_time=None, do_transposition=False):
        """Initializes the Terminator AI for the HEX game.
        Args:
            max_depth (int): max search depth
            heuristic_evaluator (function): heuristic evaluation function (args hex_board, maximiser_color)
            use_suggested_heuristic (bool): use the suggested heuristic function self.suggested_sum_scores
                note: overrides depth weighting
            depth_weighting (float): see alpha_beta
            random_seed (int): random seed for random component (if applicable); use 'random' for random
            do_iterative_deepening (bool): whether to perform iterative deepening
            max_time (float or None >> seconds): how much time we allow the algorithm to run for if iterative deepening is used.
                note: value overriden if do_iterative_deepening is false. Will always return at least depth 1 search.
                note: this is a soft bound. It will not cut execution while an iteration of max_depth is in progress.
            do_transposition: whether to compute depthwise transposition tables and use these to improve alpha-beta move ordering
                note: requires iterative deepening
        """
        # self.board = board
        self.max_depth = max_depth
        self.depth_weighting = depth_weighting
        if random_seed != 'random':
            self.random_seed = random_seed
        else:
            self.random_seed = int(random.random() * 10000)
        if use_suggested_heuristic:
            self.heuristic_evaluator = self.suggested_sum_scores
            self.depth_weighting = 0.5
        else:
            self.heuristic_evaluator = heuristic_evaluator

        if do_iterative_deepening:
            self.do_iterative_deepening = True
            self.max_time = max_time
            self.do_transposition = do_transposition
        else:
            self.do_iterative_deepening = False
            self.max_time = None
            self.do_transposition = False

        self.last_seen_num_tiles = 0  # how many tiles were colored in the last-received board, i.e. determine number of moves we are 'behind'
        self.transposition_table = None
        if self.do_transposition:
            self.transposition_table = [{} for _ in
                                        range(self.max_depth + 1)]  # initialise a list of level-wise transpositions

    def terminator_move(self, board, do_tile_increase=True):
        """Returns the best move according to the chosen heuristic evaluation function an the min-max algorithm.
        Args:
            board (HexBoard): The current hex board.
            do_tile_increase (bool): Whether to increase self.last_seen_num_tiles after this move.
        Returns:
            (int, int): AI player move.
        """

        if not self.do_iterative_deepening:
            old_state = random.getstate()  # state capture
            move = self.terminator_min_max(board, self.max_depth, 'max')  # get move
            random.setstate(old_state)  # state restore
            if do_tile_increase:  # tile counting
                num_tiles_now = 0
                for k in board.board.keys():
                    if board.board[k] != board.EMPTY:
                        num_tiles_now += 1
        else:
            # tranposition culling: cull unneeded levels from the table
            num_tiles_now = 0
            for k in board.board.keys():
                if board.board[k] != board.EMPTY:
                    num_tiles_now += 1
            if self.do_transposition:
                self.cull_transposition_table(num_tiles_now - self.last_seen_num_tiles)

            for depth in range(1, self.max_depth + 1):
                start_time = time.time()  # move timing
                old_state = random.getstate()  # state capture
                move = self.terminator_min_max(board, depth, 'max')  # get move
                random.setstate(old_state)  # state restore
                end_time = time.time()
                if self.max_time != None:
                    if (end_time - start_time) > self.max_time:
                        break  # break upon time exceed

        if do_tile_increase:
            self.last_seen_num_tiles = num_tiles_now + 1
        return move

    def random_move(self, board):
        """Returns random, free move
        Args:
            board (HexBoard): The current hex board.
        Returns:
            (int, int): AI player move.
        """
        while True:
            x = random.randint(0, board.board_size - 1)
            y = random.randint(0, board.board_size - 1)
            move = board.coord_to_string((x, y))
            if move not in board.move_list:
                new_move = board.is_empty((x, y))
                return x, y

    def terminator_min_max(self, hex_board, depth, max_or_min):
        """Returns the best position according to the min-max algorithm.
        Args:
            hex_board (HexBoard): The current hex board.
            depth (int): Search depth of the min-max algorithm.
            max_or_min (str): maximise or minimise the current color to play (hex_board.blue_to_move). Is either 'min' or 'max'.
        Returns:
            (int, int): AI player move.
        """
        alpha = float('-inf')  # initial alpha beta bounds
        beta = float('inf')
        TT_offset = 0
        if self.do_transposition:
            TT_offset = self.max_depth - depth # level offset in the transpostion table. This is necessary when using iterative deepening
            moves = order_moves_TT(hex_board, max_or_min, self.transposition_table[depth + TT_offset])
        else:
            moves = hex_board.get_free_positions()

        is_game_over = False
        if hex_board.check_win(hex_board.BLUE) or hex_board.check_win(hex_board.RED):
            is_game_over = True

        if depth == 0 or is_game_over or len(moves) == 0:  # leaf node or game over
            print("@TerminatorHex.terminator_min_max: no valid moves left, or out of depth")
            return None  # shouldn't happen
        elif max_or_min == 'max':  # maximise
            value = float('-inf')
            best_move = moves[0]  # prevent None return
            for move in moves:
                deepened_board = HexBoard(hex_board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=hex_board.move_list)
                deepened_board.set_position_auto(move)
                #print(deepened_board.move_list)
                #print(hex_board.move_list)
                # new_value = minimax(deepened_board, depth - 1, 'min', self.heuristic_evaluator) # use for minimax
                new_value, alpha, beta = alpha_beta(deepened_board, depth - 1, 'min', alpha, beta,
                                                    self.heuristic_evaluator, depth_weighting=self.depth_weighting,
                                                    transposition_table=self.transposition_table, TT_offset=TT_offset)
                if (new_value > value):
                    value = new_value
                    best_move = move
                    if self.do_transposition:
                        self.transposition_table[depth + TT_offset][board_as_hash_key(hex_board)] = value
            return best_move
        elif max_or_min == 'min':  # minimise
            value = float('inf')
            best_move = moves[0]
            for move in moves:
                deepened_board = HexBoard(hex_board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=hex_board.move_list)
                deepened_board.set_position_auto(move)
                # new_value = minimax(deepened_board, depth - 1, 'max', self.heuristic_evaluator)
                new_value, alpha, beta = alpha_beta(deepened_board, depth - 1, 'max', alpha, beta,
                                                    self.heuristic_evaluator, depth_weighting=self.depth_weighting,
                                                    transposition_table=self.transposition_table, TT_offset=TT_offset)
                if (new_value < value):
                    value = new_value
                    best_move = move
                    if self.do_transposition:
                        self.transposition_table[depth + TT_offset][board_as_hash_key(hex_board)] = value
            return best_move

    def suggested_sum_scores(self, hex_board, maximiser_color):
        val = dijkstra_score_heuristic(hex_board, maximiser_color, max_score=(hex_board.board_size * 10))
        val += 0.6 * board_center_control(hex_board, maximiser_color)
        val += 0.2 * random.random()
        return val

    def cull_transposition_table(self, num):
        """Clears levels from the transposition table. Call after making a move(s).
            Args:
                num (int, positive): how many levels to cull (max = self.max_depth).
        """
        if not self.do_transposition:
            print("@TerminatorHex.cull_transposition_table: cull called, but we're not in transposition mode")
            return

        num_real = max(0, min(self.max_depth + 1, num))
        for i in range(num_real):
            self.transposition_table.pop(self.max_depth)  # reset level dict at top level
            self.transposition_table.insert(0, {})  # append empty level table
            # note: optimise to deque

def random_score_heuristic(board, maximiser_color):
    """Returns a random score between -10 and 10"""
    return random.randint(-10, 10)


def minimax(hex_board, depth, max_or_min, evaluator):
    """The minimax algorithm on the HexBoard
        Args:
            hex_board (HexBoard): The current hex board.
            depth (int): maximum depth to search
            max_or_min (str): maximise or minimise the current color to play (hex_board.blue_to_move). Is either 'min' or 'max'.
            evaluator (function): evaluator function. Called with args hex_board, maximiser_color
        Returns:
            int: maximised/minimised value according to the evaluator
        """
    moves = hex_board.get_free_positions()
    maximiser_color = [hex_board.BLUE, hex_board.RED][(max_or_min == 'max') ^ (hex_board.blue_to_move)]
    is_game_over = False
    if hex_board.check_win(hex_board.BLUE) or hex_board.check_win(hex_board.RED):
        is_game_over = True

    # minimax:
    if depth <= 0 or is_game_over or len(moves) == 0:  # end state
        return (evaluator(hex_board, maximiser_color))
    elif max_or_min == 'max':  # maximise
        value = float('-inf')
        for move in moves:
            deepened_board = HexBoard(board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=board.move_list)
            deepened_board.set_position_auto(move)
            new_value = minimax(deepened_board, depth - 1, 'min', evaluator)
            value = [value, new_value][new_value > value]
        return value
    elif (max_or_min == 'min'):  # minimise
        value = float('inf')
        for move in moves:
            deepened_board = HexBoard(board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=board.move_list)
            deepened_board.set_position_auto(move)
            new_value = minimax(deepened_board, depth - 1, 'max', evaluator)
            value = [value, new_value][new_value < value]
        return value

    print("@minimax: unknown max_or_min objective", max_or_min)
    return None


def alpha_beta(hex_board, depth, max_or_min, alpha, beta, evaluator, depth_weighting=0, transposition_table=None, TT_offset=0):
    """The minimax algorithm on the HexBoard with alpha-beta-pruning
        Args:
            hex_board (HexBoard): The current hex board.
            depth (int): maximum depth to search
            max_or_min (str): maximise or minimise the current color to play (hex_board.blue_to_move). Is either 'min' or 'max'.
            alpha, beta (int): initial alpha and beta bounds. Can be float('-inf') and float('inf')
            evaluator (function): evaluator function. Called with args hex_board, maximiser_color
            depth-weighting (float): add heuristic score weight to the current evaluation depth. This can be used to
                force immediate capitalisation on good moves
            transposition_table: use transposition table. Enter None to disable transposition table usage.
            TT_offset (int): level index offset in the transpostion table. This is necessary when using iterative deepening,
                i.e. when a shallower than TerminatorHex.max_depth search is performed
        Returns:
            int: maximised/minimised value according to the evaluator
    """
    maximiser_color = [hex_board.BLUE, hex_board.RED][(max_or_min == 'max') ^ (hex_board.blue_to_move)]
    is_game_over = False
    if (hex_board.check_win(hex_board.BLUE) or hex_board.check_win(hex_board.RED)):
        is_game_over = True

    use_transposition = [False, True][transposition_table != None]
    if depth > 0:
        if use_transposition:
            moves = order_moves_TT(hex_board, max_or_min, transposition_table[depth + TT_offset])
        else:
            moves = hex_board.get_free_positions()
    else:
        moves = None  # don't need to compute this, save time

    # minimax with alpha-beta pruning:
    if (depth <= 0 or is_game_over or len(moves) == 0):  # end state
        value = evaluator(hex_board, maximiser_color) + (depth_weighting * depth)
        if use_transposition:
            transposition_table[depth][board_as_hash_key(hex_board)] = value
        return (value, alpha, beta)
    elif (max_or_min == 'max'):  # maximise
        value = float('-inf')
        for move in moves:
            deepened_board = HexBoard(hex_board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=hex_board.move_list)
            deepened_board.set_position_auto(move)
            new_value, _, _ = alpha_beta(deepened_board, depth - 1, 'min', alpha, beta, evaluator,
                                         depth_weighting=depth_weighting, transposition_table=transposition_table,
                                         TT_offset=TT_offset)
            if new_value > value:
                value = new_value
                if use_transposition:
                    transposition_table[depth + TT_offset][board_as_hash_key(hex_board)] = value
            alpha = [alpha, new_value][new_value > alpha]
            if alpha >= beta:  # beta cutoff
                # print("beta", alpha, beta)
                break
        return (value, alpha, beta)
    elif (max_or_min == 'min'):  # minimise
        value = float('inf')
        for move in moves:
            deepened_board = HexBoard(hex_board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=hex_board.move_list)
            deepened_board.set_position_auto(move)
            new_value, _, _ = alpha_beta(deepened_board, depth - 1, 'max', alpha, beta, evaluator,
                                         depth_weighting=depth_weighting, transposition_table=transposition_table,
                                         TT_offset=TT_offset)
            if new_value < value:
                value = new_value
                if use_transposition:
                    transposition_table[depth + TT_offset][board_as_hash_key(hex_board)] = value
            beta = [beta, new_value][new_value < beta]
            if alpha >= beta:  # alpha cutoff
                # print("alpha", alpha, beta)
                break
        return (value, alpha, beta)

    print("@alpha_beta: unknown max_or_min objective", max_or_min)
    return None


def order_moves_TT(hex_board, max_or_min, transposition_table, return_key_values=False):
    """Used with the transposition table algorithm. If we have a transposition table bound to self,
            return a move list for the current board sorted by heuristic in the TT.
            Relies on hex_board.blue_to_move for color determination.
        Args:
            hex_board (HexBoard): Hex board to evaluate
            max_or_min ('min' or 'max'): Sort by minimum eval values or maximum respectively.
            transposition_table: a dict-form transposition table
            return_key_values (bool): return (move, value) pairs instead of list of moves.
        Returns:
            list: Sorted list of moves
    """
    moves = [[move, 0] for move in hex_board.get_free_positions()]  # [move, score]
    for m in range(len(moves)):
        move = moves[m][0]
        deepened_board = HexBoard(hex_board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=hex_board.move_list)
        deepened_board.set_position_auto(move)
        try:
            moves[m][1] = transposition_table[board_as_hash_key(deepened_board)]
        except KeyError:
            # print("@order_moves_TT: could not find deepened board in transposition table. Move order may now be incorrect") # actually, this is expected behaviour
            moves[m][1] = [float('inf'), float('-inf')][
                max_or_min == 'max']  # instead, append it to the end of the moves list, do the known moves first
    sort_order = [False, True][max_or_min == 'max']  # so default = min
    moves.sort(key=lambda val: val[1], reverse=sort_order)  # sort ascending or descending
    if return_key_values:
        return moves
    else:
        return [val[0] for val in moves]  # handle from 0 to len as proper ordering


def board_hash(hex_board, maximiser_color):
    """Hash the board, for deterministic random AI position evaluator.
    Args:
        hex_board (HexBoard): The current hex board.
        maximiser_color (HexBoard color): Provided by TerminatorHex, unused in this function.
    Returns:
        int: Hash value for the current board.
    """
    board_list = []
    for k in hex_board.board.keys():
        board_list.append(hex_board.board[k])
    return hash(tuple(board_list))


def board_as_hash_key(hex_board):
    """Same as board_hash, but does not return hash value, only tuple key.
    Args:
        hex_board (HexBoard): The board to hash.
    Returns:
        tuple: Hashable unique representation of the board.
    """
    board_list = []
    for k in hex_board.board.keys():
        board_list.append(hex_board.board[k])
    return tuple(board_list)

def dijkstra_score_heuristic(hex_board, maximiser_color, max_score='inf'):
    """Dijkstra Hex score heuristic
        Args:
            hex_board: the board to evaluate
            maximiser_color: the color of the player to maximise
            max_score: the maximum scores returned. Use 'inf' for float('inf') float('-inf') resp.,
        Returns:
            integer: the advantage of maximiser on the current board,
            i.e. the difference in path length between your opponent and you
            maximise this quanitity
            always float('inf') for winning moves, and float('-inf') for losing moves"""

    win_return = [max_score, float('inf')][max_score == 'inf']
    loss_return = [-1 * max_score, float('-inf')][max_score == 'inf']
    opponent_color = [hex_board.BLUE, hex_board.RED][maximiser_color == hex_board.BLUE]
    my_dijkstra = board_dijkstra(hex_board, maximiser_color)
    your_dijkstra = board_dijkstra(hex_board, opponent_color)
    if your_dijkstra == 0:  # losing move
        return loss_return
    if my_dijkstra == 0:  # winning move
        return win_return
    return (your_dijkstra - my_dijkstra)  # difference in advantage: maximise this quantity


def board_dijkstra(hex_board, maximiser_color):
    """Dijkstra's algorithm on the Hex board. Compute the shortest path length of edge-edge traversal
    i.e. the number of tiles to place from edge to edge
    Args:
        hex_board (HexBoard): the board to evaluate
        maximiser_color (int): the color of the player to maximise
    Returns:
        int: The number of tiles left to place on the shortest edge-edge traversal. float('inf') if impossible.
    """

    opponent_color = [hex_board.BLUE, hex_board.RED][maximiser_color == hex_board.BLUE]
    edge_positions = []  # get edge positions to start with
    opposite_edge_positions = []  # opposite edge
    for i in range(hex_board.board_size):
        if maximiser_color == hex_board.BLUE:
            edge_positions.append((0, i))
            opposite_edge_positions.append((hex_board.board_size - 1, i))
        else:
            edge_positions.append((i, 0))
            opposite_edge_positions.append((i, hex_board.board_size - 1))

    minimal_traverse_path_length = float('inf')
    for e in edge_positions:
        distances = {key: float('inf') for key in hex_board.board.keys()}  # initiate distances
        if (hex_board.board[
            e] != opponent_color):  # continuing if this is true is useless because opponent has captured this edge tile
            unvisited = [k for k in hex_board.board.keys()]  # prevent node revisiting
            distances[e] = [0, 1][hex_board.board[
                                      e] == hex_board.EMPTY]  # if it is empty, set cost to 1 because we still need to fill that tile

            while (len(unvisited) > 0):
                hex_to_consider = min(unvisited, key=lambda tile: distances[tile])
                unvisited.remove(hex_to_consider)
                curr_cost = distances[hex_to_consider]
                if (distances[hex_to_consider] == float('inf')):  # unreachable components
                    break

                neighbours = hex_board.get_neighbors(hex_to_consider)
                for neigh in neighbours:
                    move_cost = 0  # assuming it is of the right color
                    if (hex_board.board[neigh] == opponent_color):
                        move_cost = float('inf')  # intraversible tile
                    elif (hex_board.board[neigh] == hex_board.EMPTY):
                        move_cost = 1  # still need to color
                    alternative_cost = curr_cost + move_cost

                    if (alternative_cost < distances[neigh]):
                        distances[neigh] = alternative_cost  # Dijkstra step

        for opedge in opposite_edge_positions:
            minimal_traverse_path_length = min(minimal_traverse_path_length,
                                               distances[opedge])  # edge-to-edge minimal distance

    return minimal_traverse_path_length


def board_center_control(hex_board, maximiser_color):
    maximiser_centrality = 0
    maximiser_num_tiles = 0
    opponent_centrality = 0
    opponent_num_tiles = 0
    bs = hex_board.board_size
    for k in hex_board.board.keys():
        if hex_board.board[k] == maximiser_color:  # my color
            maximiser_num_tiles = + 1
            maximiser_centrality += ((bs / 2) - 0.5) - max(abs((bs / 2) - k[0] - 0.5),
                                                   abs((bs / 2) - k[1] - 0.5))  # board centrality of that tile
        elif hex_board.board[k] != hex_board.EMPTY:  # opponent color
            opponent_num_tiles = + 1
            opponent_centrality += ((bs / 2) - 0.5) - max(abs((bs / 2) - k[0] - 0.5), abs((bs / 2) - k[1] - 0.5))

    maximiser_centrality /= (bs / 2)  # normalise
    opponent_centrality /= (bs / 2)
    maximiser_num_tiles = max(1, maximiser_num_tiles) # prevent d.b.z.
    opponent_num_tiles = max(1, opponent_num_tiles)
    return ((maximiser_centrality / maximiser_num_tiles) - (
            opponent_centrality / opponent_num_tiles))  # difference in center control


if __name__ == '__main__':
    import HexBoard as hb

    h = hb.HexBoard(4)
    print(board_dijkstra(h, h.BLUE))
    h.set_position((0, 0), h.BLUE)
    print(board_center_control(h, h.BLUE))
    print(board_dijkstra(h, h.BLUE))
    h.set_position((2, 1), h.BLUE)
    print(board_dijkstra(h, h.BLUE))
    h.set_position((3, 3), h.BLUE)
    print(board_dijkstra(h, h.BLUE))
    print(board_center_control(h, h.BLUE))
