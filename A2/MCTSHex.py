from HexBoard import *
import math
import random
import numpy as np

# TODO: enhancements
# formulaic node expansion strategy DONE
# frequent-visitor node re-expansion strategy DONE
# rollout move strategies: low-level alpha-beta, local heuristic
# nodes per level function: memory tracing DONE
# recursive deletion in MCTSNode?
# WinScan: scan for winning moves, don't expand further than those nodes DONE
# top-level full-scan enhancement: ensure that top-level nodes are always scanned DONE

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
        track_expansion (bool): whether to track how many child nodes can be expanded from this node.
            Used for enh_FreqVisitor
        if track_expansion:
            n_expandables (int): how many possible child nodes there are
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
        c_explore (float): the UCT formula exploration parameter
        expansion_fraction (see below): a way of determining the fraction of valid moves to expand in the tree
            Options:
            ('constant', float): the constant fraction of valid moves to expand in the child expansion step, rounded up to nearest int
            ('sigmoid', float): a 1-bounded (quasi-)sigmoid, increasing to 1 towards 0 possible moves. float controls the bound slope.
                NOTE: formula: (1 - ((n_moves - 1) / (n_moves + float))) --> check on desmos.com for clarity
            ('lambda', Callable): provide your own function. Must have one input parameter (n_moves) and one output (fraction to expand).
                NOTE: this is the fraction of moves to expand, not the number of actual moves to expand itself!
        random_seed (int or "random"): a random seed, RNG state ('random' module) is restored after every MCTS_move
        enh_WinScan (bool): enable WinScan enhancement. This enhancement scans for winning(/losing) moves in the MCTS_expand function.
            If it finds one, this move is set as the only child node, preventing further futile exploration from the parent node.
        enh_FreqVisitor (bool): enable Frequent Visitor enhancement. This enhancement is useful when a reduced expansion fraction is used.
            It gradually expands more child nodes for nodes that are frequently passed to increase exploration.
            This happens in the MCTS_select function.
        enh_EnsureTopLevelExplr (bool): enable Top Level Equalised Exploration enhancement. This enhancement ensures that all actual moves
            (which are top-level nodes, one below the root) are explored equally. This may have both downsides and upsides.
    """
    def __init__(self, N_trials, c_explore, expansion_function=('constant', 1), random_seed="random", enh_WinScan=False, enh_FreqVisitor=False,
                 enh_EnsureTopLevelExplr=False):
        self.N_trials = N_trials
        self.c_explore = c_explore
        
        if type(expansion_function) != tuple and len(expansion_function) != 2:
            raise Exception("@MCTSHex.__init__: invalid expansion_function parameter")
        if expansion_function[0] == "constant":
            if expansion_function[1] <= 0 or expansion_function[1] > 1:
                raise Exception("@MCTSHex.__init__: invalid expansion_function parameter: constant fraction must be between 0 and 1")
            self.expansion_function = lambda n_moves: expansion_function[1]
        elif expansion_function[0] == "sigmoid":
            if expansion_function[1] <= 0:
                raise Exception("@MCTSHex.__init__: invalid expansion_function parameter: sigmoid parameter too small")
            self.expansion_function = lambda n_moves: 1 - ((n_moves - 1) / (n_moves + expansion_function[1])) 
        elif expansion_function[0] == "lambda":
            self.expansion_function = expansion_function[1]
            
        self.random_seed = random_seed
        self.rollout_strategy = "random" # no options for now
        self.enh_WinScan = enh_WinScan
        self.enh_FreqVisitor = enh_FreqVisitor
        self.enh_EnsureTopLevelExplr = enh_EnsureTopLevelExplr

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
        curr_last_move = None,
        if len(board.move_list) > 0:
            curr_last_move = board.string_to_coord(board.move_list[-1])
            
        if cull_tree == True: # cull the previous MCTS tree
            self.tree_head = MCTSNode(curr_last_move, [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
            # so the root becomes the last seen move, with corresponding color
        else: # attempt to find the current state in the previous MCTS tree
            if self.tree_head == None or board.move_list == None or board.move_list == [] or self.previous_board == None: # tree doesn't exist or isn't useful
                self.tree_head = MCTSNode(curr_last_move, [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
            else:
                if self.previous_board.move_list != [board.string_to_coord(mv) for mv in board.move_list[0:len(self.previous_board.move_list)]]: # non-matching boards, cull tree
                    self.tree_head = MCTSNode(curr_last_move, [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
                else: # matching boards
                    for move in board.move_list[len(self.previous_board.move_list):]: # for all new moves
                        matching_child = None
                        for child in self.tree_head.children:
                            if child.move == board.string_to_coord(move):
                                matching_child = child
                        self.tree_head = matching_child

                        if matching_child == None: # no matching child node found, cull tree
                            self.tree_head = MCTSNode(curr_last_move, [HexBoard.BLUE, HexBoard.RED][color == HexBoard.BLUE])
        
        old_state = random.getstate() # RNG mechanism: load old state after finishing move
        if self.random_seed != "random":
            random.seed(self.random_seed)

        for t in range(self.N_trials): # do N_trials rollouts
            copy_board = HexBoard(board.board_size, n_players=2, enable_gui=False, interactive_text=False, ai_move=None, blue_ai_move=None, red_ai_move=None, move_list=board.move_list)
            node_to_explore = self.tree_head # by default, explore from root
            if self.enh_EnsureTopLevelExplr: # force exploration of the top-level move nodes, one below the root, i.e. all valid moves
                if self.tree_head != None and self.tree_head.children != []: # not non-existent or a leaf node
                    list_equality = np.array([node.n_trials for node in self.tree_head.children])
                    dev_equality = list_equality - np.average(list_equality) # deviations from how many average trials per top-level node there are
                    minimums = list(np.where(dev_equality == np.amin(dev_equality))[0]) # all indices where exploration is minimum
                    index_for_exploration = random.choice(minimums) # pick a random top-level minimally explored node to explore
                    node_to_explore = self.tree_head.children[index_for_exploration] # explore from that node instead
                    copy_board.set_position_auto(node_to_explore.move) # update position on board
                
            path, deepened_board, check_win = self.MCTS_select(copy_board, node_to_explore, []) # path to leaf node
            if self.enh_EnsureTopLevelExplr: # add the root node to the path in case of lower level start
                path.insert(0, self.tree_head)
                
            if check_win == None: # the leaf node is not a winning situation
                node_to_rollout = random.choice(path[-1].children) # randomly choose a child of the leaf node to rollout
                path.append(node_to_rollout)
                deepened_board.set_position_auto(node_to_rollout.move) # do the child move
                winning_color = self.MCTS_rollout(deepened_board, node_to_rollout) # perform one rollout, get the winning color
                self.MCTS_backpropagate(path, winning_color) # backpropagate the win along the path
            else: # handle the leaf node win as a rollout
                self.MCTS_backpropagate(path, check_win)
        
        best_move_win_prop = -1 # always get a move
        best_move = None
        reservoir_count = 1 # used in reservoir sampling to replace best move
        for potential_move in self.tree_head.children:
            ratio = 0
            if potential_move.n_trials > 0: # avoid div by zero
                ratio = potential_move.n_wins / potential_move.n_trials
            #print("Bestmove", potential_move.move, ratio)
            if ratio >= best_move_win_prop: # better move
                if ratio == best_move_win_prop:
                    reservoir_count += 1
                    if random.random() < (1 / reservoir_count): # reservoir sampling to replace the move
                        best_move = potential_move # ATTN: this is an MCTSNode
                else: # greater
                    best_move = potential_move # ATTN: this is an MCTSNode
                    best_move_win_prop = ratio
                    reservoir_count = 1 # reset the reservoir sampling count
        # NOTE: the reservoir sampling here and in child_select is used to uniformly sample
        # moves with the same score, instead of biased or deterministic selection

        #print_MCTSNode_structure(self.tree_head) # reveal tree structure
        
        if move_head:
            self.tree_head = best_move # update the tree head ATTN: does the rest of the tree go out of scope? i.e. garbage collection
            deepened_board = self._copy_and_move(board, best_move.move)
            self.previous_board = deepened_board # update board to have matching move
        else:
            self.previous_board = board # board at current root

        random.setstate(old_state) # restore previous RNG state

        return best_move.move
        
    def MCTS_select(self, board, node, path):
        # select child nodes until leaf reached, save node path taken
        path.append(node)
        if len(node.children) == 0: # leaf node
            check_win = board.get_winning_color() # check for a winning position
            if check_win == None: # no win --> expand
                self.MCTS_expand(board, node)
            # else, return, and include the winning color in the return for handling --> count as rollout
            return path, board, check_win
        else: # not a leaf node, select child
            if self.enh_FreqVisitor: # frequent visitor enhancement: upgrade number of child nodes: randomly add one extra child node
                possible_moves = board.get_free_positions()
                if len(node.children) < len(possible_moves): # node not yet fully expanded?
                    color = [HexBoard.RED, HexBoard.BLUE][board.blue_to_move]
                    for child in node.children:
                        possible_moves.remove(child.move) # remove taken moves
                    move_pick = random.choice(possible_moves) # randomly add one move to the children set
                    node.addChildren(MCTSNode(move_pick, color))
                    
            selection = self.child_select(board, node, node.n_trials) # select the best child according to the UCT formula
            deepened_board = self._copy_and_move(board, selection.move)
            
            return self.MCTS_select(deepened_board, selection, path) # recursive

    def child_select(self, board, node, parent_node_trials):
        # do UCT formula
        ln = math.log([1, parent_node_trials][parent_node_trials > 0]) # avoid log domain error by min-capping to 1, base = math.e
        highest_UCT = -1
        best_child = None # kinda unethical, picking a best child
        reservoir_count = 1 # used in counting for reservoir sampling
        for child in node.children:
            denom = [1, child.n_trials][child.n_trials > 0] # avoid division by zero
            UCT_score = (child.n_wins / denom) + self.c_explore * math.sqrt(ln / denom)
            if UCT_score >= highest_UCT:
                if UCT_score == highest_UCT: # exact equality can occur in initial iterations
                    reservoir_count += 1
                    if random.random() < (1 / reservoir_count): # reservoir sampling to replace the move
                        best_child = child
                else: # greater
                    best_child = child
                    highest_UCT = UCT_score
                    reservoir_count = 1 # reset reservoir sampling

        return best_child

    def MCTS_expand(self, board, node):
        # expansion strategy for a node
        possible_moves = board.get_free_positions()
        exp_fraction = self.expansion_function(len(possible_moves)) # use the expansion function callable (see __init__)
        n_to_draft = int(math.ceil(exp_fraction * len(possible_moves))) # how many moves to pick for expansion
        to_expand = random.sample(possible_moves, n_to_draft) # pick n_to_draft moves from list of possible moves
        # ATTN: sample without replacement!
        color = [HexBoard.RED, HexBoard.BLUE][board.blue_to_move] # determine color to move
        
        if self.enh_WinScan == True: # WinScan enhancement
            for move in possible_moves: # here, we ignore the expansion fraction because we want to find all winning moves
                deepened_board = self._copy_and_move(board, move)
                check_win = deepened_board.get_winning_color()
                if check_win == color: # found a win, clear all child nodes, set winning node as only child
                    node.children = []
                    node.addChildren(MCTSNode(move, color))
                    break # break out of the rest of the loop
                
        if self.enh_WinScan and node.children != []: # indicating that WinScan delivered a winning move
            return
        else:
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

def get_MCTSNode_level_table(node, node_table=[], depth=0):
    """
    Recursively determine number of nodes at each depth starting from an MCTSNode root node.
    Args:
        node (MCTSNode): the root node
        node_table (list): the table for recursion. Leave as [].
        depth (int): current depth. Leave as 0.
    Returns:
        (list(int)): depth-wise node number table
    """

    if len(node_table) <= depth:
        node_table.append(0) # depth-based table, increasing index is increasing depth
    node_table[depth] += 1
    for child in node.children:
        node_table = get_MCTSNode_level_table(child, node_table=node_table, depth=depth+1)
        
    return node_table

def print_MCTSNode_structure(node, depth=0):
    min_string = "".join(["-" for _ in range(depth)])
    print(min_string, node.color, node.move, node.n_wins, node.n_trials)
    for child in node.children:
        print_MCTSNode_structure(child, depth=depth+1)
    return
