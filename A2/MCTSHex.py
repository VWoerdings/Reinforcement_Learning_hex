from HexBoard import *
import math
import random

# TODO:
# formulaic node expansion strategy
# frequent-visitor node re-expansion strategy
# rollout move strategies: low-level alpha-beta, local heuristic
# nodes per level function: memory tracing
# recursive deletion in MCTSNode?

class MCTSNode:
    """
    A node in the MCTS tree for Hex.
    Fields:
        move (int, int): the corresponding move on the board
            NOTE: to improve memory efficiency, the board itself is not stored
        color (HexBoard color): the color of the tile move
        children (list(MCTSNode)): child MCTSNodes
        n_wins (int): number of rollout wins from this node
        n_trials (int): number of rollout trials from this node
    """
    
    def __init__(self, move, color):
        self.move = move
        self.color = color
        self.children = []

        self.n_wins = 0
        self.n_trials = 0

    def addChildren(self, new_children):
        self.children.append(new_children)
        return

    def processWin(self, color):
        # backpropagate a win
        self.n_trials += 1
        if color == self.color: # my color won
            self.n_wins += 1
        return
            
class MCTSHex:
    """
    The Monte-Carlo Tree Search Hex game AI implementation.
    Args:
        N_trials (int): number of trials per tree search cycle
        c_explore (float, [0 to 1]): the UCT formula exploration parameter
        expansion_fraction (float, [0 to 1]): the fraction of valid moves to expand in the child expansion step, rounded up to nearest int
        random_seed (int or "random"): a random seed, RNG state ('random' module) is restored after every MCTS_move
    """
    def __init__(self, N_trials, c_explore, expansion_fraction=1, random_seed="random"):
        self.N_trials = N_trials
        self.c_explore = c_explore
        self.expansion_fraction = expansion_fraction
        self.random_seed = random_seed
        self.rollout_strategy = "random" # no options for now

        self.tree_head = None
        self.previous_board = None
        
    def MCTS_move(self, board, cull_tree=False, move_head=True): # retrieve a MCTS move
        """
        Perform the MCTS algorithm and return a valid move.
        Args:
            board (HexBoard): the current board state
            cull_tree (bool): whether to cull the entire MCTS tree initially, or attempt to find the current board state in the existing tree
            move_head (bool): whether to move the tree head node downwards to the selected node automatically after MCTS_move
        Returns:
            (int, int): the chosen move on the HexBoard
        """
        
        color = [HexBoard.RED, HexBoard.BLUE][board.blue_to_move] # determine color to move
        if cull_tree == True: # cull the previous MCTS tree
            self.tree_head = MCTSNode(board.move_list[-1], [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
            # so the root becomes the last seen move, with corresponding color
        else: # attempt to find the current state in the previous MCTS tree
            if self.tree_head == None: # tree doesn't exist
                self.tree_head = MCTSNode(board.move_list[-1], [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
                
            if previous_board.move_list != board.move_list[0:len(previous_board.move_list)]: # non-matching boards, cull tree
                self.tree_head = MCTSNode(board.move_list[-1], [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
            else: # matching boards
                for move in board.move_list[len(previous_board.move_list):]: # for all new moves
                    matching_child = None
                    for child in self.tree_head.children:
                        if child.move == move:
                            matching_child = child
                    self.tree_head = matching_child

                    if matching_child == None: # no matching child node found, cull tree
                        self.tree_head = MCTSNode(board.move_list[-1], [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
        
        old_state = random.getstate() # RNG mechanism: load old state after finishing move
        if self.random_seed != "random":
            random.seed(self.random_seed)

        for _ in range(N_trials): # do N_trials rollouts
            path, deepened_board, check_win = self.MCTS_select(board, self.tree_head, []) # path to leaf node
            if check_win == None: # the leaf node is not a winning situation
                node_to_rollout = random.choice(path[-1].children) # randomly choose a child of the leaf node to rollout
                deepened_board.set_position_auto(node_to_rollout.move) # do the child move
                winning_color = self.MCTS_rollout(deepened_board, node_to_rollout) # perform one rollout, get the winning color
                self.MCTS_backpropagate(path, winning_color) # backpropagate the win along the path
            else: # handle the leaf node win as a rollout
                self.MCTS_backpropagate(path, check_win)

        best_move_win_prop = -1 # always get a move
        best_move = None
        for potential_move in self.tree_head.children:
            if (potential_move.n_wins / potential_move.n_trials) > best_move_win_prop: # better move
                best_move = potential_move # ATTN: this is an MCTSNode
                best_move_win_prop = (potential_move.n_wins / potential_move.n_trials)

        if move_head:
            self.tree_head = best_move # update the tree head ATTN: does the rest of the tree go out of scope? i.e. garbage collection
            deepened_board = self._copy_and_move(board, best_move)
            self.previous_board = deepened_board # update board to have matching move
        else:
            self.previous_board = board # board at current root

        random.setstate(old_state) # restore previous RNG state

        return best_move.move
        
    def MCTS_select(self, board, node, path):
        # select child nodes until leaf reached, save node path taken
        if len(node.children == 0): # leaf node
            path.append(node)
            check_win = board.get_winning_color() # check for a winning position
            if check_win == None: # no win --> expand
                MCTS_expand(board, node)
            # else, return, and include the winning color in the return for handling --> count as rollout
            return path, board, check_win
        else: # not a leaf node, select child
            selection = self.child_select(board, node)
            path.append(selection)
            deepened_board = self._copy_and_move(board, selection.move)
            
            return MCTS_select(deepened_board, selection, path) # recursive

    def child_select(self, board, node, parent_node_trials):
        # do UCT formula
        ln = math.log(parent_node_trials) # base = math.e
        highest_UCT = -1
        best_child = None # kinda unethical, picking a best child
        for child in node.children:
            UCT_score = (child.wins / child.n_trials) + self.c_explore * math.sqrt(ln / child.n_trials)
            if UCT_score > highest_UCT:
                best_child = child
                highest_UCT = UCT_score

        return best_child

    def MCTS_expand(self, board, node):
        # expansion strategy for a node
        possible_moves = board.get_free_positions()
        n_to_draft = int(math.ceil(self.expansion_fraction * len(possible_moves))) # how many moves to pick for expansion
        to_expand = random.choices(possible_moves, k=n_to_draft) # pick n_to_draft moves from list of possible moves
        color = [HexBoard.BLUE, HexBoard.RED][board.blue_to_move] # determine color to move, inverted from current board!!!
        for move in to_expand:
            new_node = MCTSNode(move, color) # create nodes for each move
            node.addChildren(new_node) # add move node to current selected node
        return

    def MCTS_rollout(self, board, node):
        # random (?) rollout from the given node with the given board
        while True: # scary
            check_win = board.get_winning_color()
            if check_win != None: # got a winner!
                break
            
            possible_moves = board.get_free_positions()
            if len(possible_moves) == 0:
                raise Exception("@MCTSHex.MCTS_rollout: something weird happened, no more moves available, but no win detected")
                return None
            
            if self.rollout_strategy == "random":
                picked_move = random.choice(possible_moves) # pick a random move
            board.set_position_auto(picked_move)

        return check_win

    def MCTS_backpropagate(self, path, winning_color):
        # propagate win along the node path
        for elem in path:
            elem.processWin(winning_color)
        return

    @classmethod
    def _copy_and_move(cls, board, move):
        # copy a HexBoard and make a move (according to board.blue_to_move)
        deepened_board = HexBoard(board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=board.move_list)
        deepened_board.set_position_auto(move)

        return deepened_board
