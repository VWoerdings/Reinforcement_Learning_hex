# from __future__ import print_function
import sys

import numpy as np
from Game import Game
from hex.HexBoard import HexBoard

sys.path.append('..')


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
        return self.n * self.n + 1  # +1 for pass

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

        if action == self.n * self.n:
            # Pass
            return board, -player

        b = HexBoard(self.n)
        b.board = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.place(move, player)
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
        valids = np.append(valids, [1])  # Append 1 for pass
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
        assert (len(pi) == self.n ** 2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
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
