# from __future__ import print_function
import sys
sys.path.append('..')

import numpy as np
from Game import Game
from hex.HexBoard import HexBoard



class HexGame(Game):
    square_content = {
        -1: "r",
        +0: "-",
        +1: "b"
    }

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board = HexBoard(self.n)
        return board.board

    def getBoardSize(self):
        return self.n, self.n

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.n * self.n

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

        b = HexBoard(self.n)
        b.board = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.place_debug(move, player)
        return b.board, -player

    def getNextState_debug(self, board, player, action, **kwargs):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

        b = HexBoard(self.n)
        b.board = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.place_debug(move, player, **kwargs)
        return b.board, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        b = HexBoard(self.n)
        b.board = np.copy(board)
        legal_moves = b.get_free_positions()
        valids = np.zeros((self.n, self.n))
        valids[legal_moves] = 1
        valids = valids.reshape(self.n ** 2)
        return valids


    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        b = HexBoard(self.n)
        b.board = np.copy(board)

        if b.check_win(player):
            return 1
        elif b.check_win(-player):
            return -1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return player * board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # mirror, rotational
        assert (len(pi) == self.n ** 2 )
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    @staticmethod
    def display(board):
        n = board.shape[0]
        """Prints the board to the console."""
        print("   ", end="")
        for y in range(n):
            print(chr(y + ord('a')), "", end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            for z in range(y):
                print(" ", end="")
            print(y, "|", end="")
            for x in range(n):
                piece = board[x, y]
                if piece == HexBoard.BLUE:
                    print("b ", end="")
                elif piece == HexBoard.RED:
                    print("r ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("   -----------------------")

if __name__ == "__main__":
    player = HexBoard.BLUE
    game = HexGame(1)
    board = game.getInitBoard()
    board, player = game.getNextState(board, player, 0)
    # board, player = game.getNextState(board,player,2)
    # board, player = game.getNextState(board,player,3)
    # board, player = game.getNextState(board,player,1)
    # board, player = game.getNextState(board,player,6)
    # board, player = game.getNextState(board,player,8)
    # board, player = game.getNextState(board,player,0)
    game.display(board)
    print(game.getValidMoves(board, player))
    test_board = HexBoard(3)
    test_board.board = np.copy(board)
    print(test_board.get_free_positions())
    print(game.getGameEnded(board,-player))