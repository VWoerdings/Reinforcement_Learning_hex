import numpy as np
import sys

class HexBoard:
    BLUE = 1
    EMPTY = 0
    RED = -1

    def __init__(self, board_size):
        """Initializes the board.
        Args:
            board_size (int): Size of the hexagon grid.
        """
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size))
        self.game_over = False

    def is_game_over(self):
        """Checks if game is over.
        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.game_over

    def is_empty(self, coordinates):
        """Checks if the board is empty at given coordinates.
        Args:
            coordinates (int, int): X and y coordinates to check.
        Returns:
            bool: True if the given position is empty, False otherwise.
        """
        return self.board[coordinates] == HexBoard.EMPTY

    def is_color(self, coordinates, color):
        """Checks if the board is a given color at given coordinates.
        Args:
            coordinates (int, int): X and y coordinates to check.
            color (int): Color to compare to.
        Returns:
            bool: True if the given position is equal to the given color, False otherwise.
        """
        return self.board[coordinates] == color

    def get_color(self, coordinates):
        """Finds the color at given coordinates.
        Args:
            coordinates (int, int): X and y coordinates to check.
        Returns:
            int: Color at the given position.
        """
        if coordinates == (-1, -1):
            return HexBoard.EMPTY
        return self.board[coordinates]

    def place(self, coordinates, color):
        """Places a given color at given coordinates.
        Args:
            coordinates (int, int): X and y coordinates to check.
            color (int): Color to place.
        """
        if self.game_over:
            print("The game is already over.")
            return
        elif self.board[coordinates] == HexBoard.BLUE or self.board[coordinates] == HexBoard.RED:
            self.print()
            print("Tried coordinates: %s" % (coordinates,))
            raise ValueError("Hex is already occupied.")
        else:
            self.board[coordinates] = color
            if self.check_win(HexBoard.BLUE) or self.check_win(HexBoard.RED):
                self.game_over = True

    def place_debug(self, coordinates, color, **kwargs):
        """Places a given color at given coordinates.
        Args:
            coordinates (int, int): X and y coordinates to check.
            color (int): Color to place.
        """
        if self.game_over:
            print("The game is already over.")
            return
        elif self.board[coordinates] == HexBoard.BLUE or self.board[coordinates] == HexBoard.RED:
            self.print()
            print("Tried coordinates: %s" % (coordinates,))
            print(kwargs)
            sys.stdout.flush()
            raise ValueError("Hex is already occupied.")
        else:
            self.board[coordinates] = color
            if self.check_win(HexBoard.BLUE) or self.check_win(HexBoard.RED):
                self.game_over = True

    @staticmethod
    def coord_to_string(coordinates):
        """Converts grid coordinates to a string.
        Args:
            coordinates (int, int): Grid coordinates to convert.
        Returns:
            str: Coordinates in string format
        Examples:
            >>>HexBoard.coord_to_string((0,0))
            'a0'
        """
        return "" + str(chr(coordinates[0] + 97)) + str(coordinates[1])

    @staticmethod
    def string_to_coord(input_string):
        """Converts string to grid coordinates.
        Args:
            input_string (str): String to convert.
        Returns:
            (int,int): Converted grid coordinates.
        Examples:
            >>>HexBoard.string_to_coord('a0')
            (0,0)
        """
        x = ord((input_string[0])) - 97
        y = int(input_string[1])
        return x, y

    @staticmethod
    def get_opposite_color(current_color):
        """Gets opposite to given color.
        Args:
            current_color (int): Input color.
        Returns:
            (int): Opposite color.
        """
        return -current_color

    def get_neighbors(self, coordinates):
        """Gets list of neighbours of a given position.
        Args:
            coordinates (int, int): Coordinates to calculate neigbours for.
        Returns:
            :obj:`list` of (int,int): List of neighbour coordinates.
        """
        (cx, cy) = coordinates
        neighbors = []
        if cx - 1 >= 0:
            neighbors.append((cx - 1, cy))
        if cx + 1 < self.board_size:
            neighbors.append((cx + 1, cy))
        if cx - 1 >= 0 and cy + 1 <= self.board_size - 1:
            neighbors.append((cx - 1, cy + 1))
        if cx + 1 < self.board_size and cy - 1 >= 0:
            neighbors.append((cx + 1, cy - 1))
        if cy + 1 < self.board_size:
            neighbors.append((cx, cy + 1))
        if cy - 1 >= 0:
            neighbors.append((cx, cy - 1))
        return neighbors

    def border(self, color, move):
        """Checks if a hexagon is located on the right (if blue) or bottom (if blue) border of the board.
        Args:
            color: (int) Color of the hexagon.
            move: (int,int) Coordinates of the hexagon.
        Returns:
            (bool): True if the hexagon is located at the border of the board.
        """
        (nx, ny) = move
        return (color == HexBoard.BLUE and nx == self.board_size - 1) or (
                color == HexBoard.RED and ny == self.board_size - 1)

    def traverse(self, color, move, visited):
        """Traverses hexagons from top to bottom (if red) or from left to right (if blue).
        Args:
            color (int): Color of the current hexagon.
            move (int,int): Coordinates of the current hexagon.
            visited (dict of (int,int):bool): Maps each set of coordinates to a bool: True if the hexagon was visited
                before.
        Returns:
            (bool): True if the current hexagon connects to a border hexagon.
        """
        if not self.is_color(move, color) or (move in visited and visited[move]):
            return False
        if self.border(color, move):
            return True
        visited[move] = True
        for n in self.get_neighbors(move):
            if self.traverse(color, n, visited):
                return True
        return False

    def check_win(self, color):
        """Checks if there is a winner.
        Returns:
            (bool): True if there is a winner, false otherwise.
        """
        for i in range(self.board_size):
            if color == HexBoard.BLUE:
                move = (0, i)
            else:
                move = (i, 0)
            if self.traverse(color, move, {}):
                return True
        return False

    def print(self):
        """Prints the board to the console."""
        print("   ", end="")
        for y in range(self.board_size):
            print(chr(y + ord('a')), "", end="")
        print("")
        print(" -----------------------")
        for y in range(self.board_size):
            for z in range(y):
                print(" ", end="")
            print(y, "|", end="")
            for x in range(self.board_size):
                piece = self.board[x, y]
                if piece == HexBoard.BLUE:
                    print("b ", end="")
                elif piece == HexBoard.RED:
                    print("r ", end="")
                else:
                    if x == self.board_size:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("   -----------------------")

    def reset_board(self):
        """Resets board."""
        self.board = np.zeros((self.board_size, self.board_size))
        self.game_over = False

    def get_free_positions(self):
        """Returns free positions on the board
        Returns:
             2D Boolean array, True where the board is empty
        """
        return self.board == HexBoard.EMPTY

if __name__ == "__main__":
    b = HexBoard(3)
    b.place((0,0),HexBoard.BLUE)
    b.place((0,1),HexBoard.RED)
    b.place((1,0),HexBoard.BLUE)
    b.place((1,1),HexBoard.RED)
    b.place((2,0),HexBoard.BLUE)
    b.print()
    print(b.game_over)
    print(b.get_free_positions())