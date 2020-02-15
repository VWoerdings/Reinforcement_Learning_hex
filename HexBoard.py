import tkinter as tk
from math import cos, tan, pi
from tkinter import messagebox
from RegularPolygon import RegularPolygon
import re
from itertools import groupby
import os
from TerminatorHex import TerminatorHex

class HexBoard:
    """A class that stores and computes the state of a hex board.
    """
    # Hexagon properties
    HEX_SIZE = 50  # Side-to-side distance
    APOTHEM = HEX_SIZE / 2  # Midpoint of hexagon to midpoint of a side distance
    SIDE_LENGTH = 2 * APOTHEM * tan(pi / 6)  # Length of a side
    CROSS_LENGTH = 2 * (APOTHEM / cos(pi / 6))  # Vertex-to-vertex distance

    # GUI properties
    BORDER_WIDTH = 3
    MIN_PADDING = 25
    MIN_PADDING_X = APOTHEM + MIN_PADDING
    MIN_PADDING_y = CROSS_LENGTH / 2 + MIN_PADDING
    X_PADDING = MIN_PADDING_X
    Y_PADDING = MIN_PADDING_y

    BLUE = 1
    RED = 2
    EMPTY = 3
    
    #Pattern match 
    PATTERN = '[a-zA-Z][0-9]'


    def __init__(self, board_size, enable_GUI=False, interactive_text=False):
        """Initializes the board and GUI if applicable.
        Args:
            board_size (int): Size of the hexagon grid.
            enable_GUI (bool, optional): Enables an interactive GUI. Default is False.
            interactive_text (bool, optional): Enables an interactive text mode. Sets enable_GUI to False. Default is False. Maximum board size for this mode is currently 10.
        """
        self.board = {}
        self.board_size = board_size
        self.game_over = False
        for x in range(board_size):
            for y in range(board_size):
                self.board[x, y] = HexBoard.EMPTY

        self.move_list = []

        self.blue_to_move = True
        self.enable_GUI = enable_GUI

        self.interactive_text = interactive_text
        self.quit = False
        if interactive_text:
            self.enable_GUI = False
            self.interactive_text_loop()

        if self.enable_GUI:
            self.WIN_WIDTH = 2 * HexBoard.X_PADDING + (self.board_size - 1) * 1.5 * HexBoard.HEX_SIZE
            self.WIN_HEIGHT = 2 * HexBoard.Y_PADDING + (self.board_size - 1) * (
                    (HexBoard.CROSS_LENGTH + HexBoard.SIDE_LENGTH) / 2)

            self.window = tk.Tk()
            self.window.wm_title("Hex")

            self.canvas = tk.Canvas(self.window, width=self.WIN_WIDTH, height=self.WIN_HEIGHT)
            self.canvas.pack()

            self.canvas.bind("<Button-1>", self.on_click)
            self.create_GUI()
            self.window.mainloop()

    def interactive_text_loop(self):
        """Contains an infinite loop that reads and handles commands from the console."""
        self.quit = False
        self.print_command_help()
        while not self.quit:
            if self.blue_to_move:
                command = input("Enter a command ")
                if valid_command(command, self.board_size): # Needs to be extended to support board sizes of > 10
                    command = list(split_text(command))
                    command[0].lower()
                    x, y = (ord(command[0])- 97), int(command[1])
                    self.place((x, y), HexBoard.BLUE)
                elif command == 'quit' or command == 'q':
                    self.quit = True
                    return
                elif command == 'print' or command == 'p':
                    self.printBroad()
                elif command == 'help' or command == 'h':
                    self.print_command_help()
                elif command == 'undo' or command == 'u':
                    self.undo_move()
                elif command == 'reset' or command == 'r':
                    self.reset_board()
                else:
                    print('Command \'' + command + '\' not recognized, please enter a valid command.')
            else:
               th = TerminatorHex(self)
               x = th.InitiateTerminator()
               self.place((x[0], x[1]), HexBoard.RED)
               self.printBroad()


    def is_valid(self, coordinates):
        """ Checks if coordinates are within the bounds of the grid.
        Args:
            coordinates (int,int): Coordinates to check.
        Returns: True if coordinates are on the board, False otherwise
        """
        return 0 <= coordinates[0] <= self.board_size - 1 and 0 <= coordinates[1] <= self.board_size - 1

    @staticmethod
    def print_command_help():
        """Prints a list of available commands for the interactive text mode
        """
        print('List of commands:')
        print('\thelp or h:\t\tPrint this overiew.')
        print('\tcoordinates:\tPlay at the given position (for example \'a0\').')
        print('\tprint or p:\t\tPrint the current state of the board.')
        print('\tquit or q:\t\tQuit the program.')
        print('\tundo or u:\t\tUndo the final move.')
        print('\treset or r:\t\tReset the game.')

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

    def check_and_print_winner(self):
        """Checks if there is a winner and prints to the GUI or console if applicable
        Returns:
            (bool): True if a winner was found, False otherwise.
        """
        if self.check_win(HexBoard.RED):
            self.game_over = True
            output_string = "The winner is RED!"
            if self.interactive_text:
                print(output_string)
            if self.enable_GUI:
                messagebox.showinfo("Window", output_string)
        elif self.check_win(HexBoard.BLUE):
            self.game_over = True
            output_string = "The winner is BLUE!"
            if self.interactive_text:
                print(output_string)
            if self.enable_GUI:
                messagebox.showinfo("Window", output_string)


    def place(self, coordinates, color):
        """Places a given color at given coordinates.
        Args:
            coordinates (int, int): X and y coordinates to check.
            color (int): Color to place.
        """
        if not self.game_over and self.board[coordinates] == HexBoard.EMPTY:
            if not self.check_and_print_winner():
                self.move_list.append(self.coord_to_string(coordinates))
                self.board[coordinates] = color
                if color == HexBoard.RED:
                    self.blue_to_move = True
                elif color == HexBoard.BLUE:
                    self.blue_to_move = False
                self.check_and_print_winner()
        elif not self.game_over and not self.board[coordinates] == HexBoard.EMPTY and color == HexBoard.EMPTY:
            self.board[coordinates] = color
        else:
            print("The game is already over.")
            os._exit(1)



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
        if current_color == HexBoard.BLUE:
            return HexBoard.RED
        return HexBoard.BLUE


    def get_neighbors(self, coordinates):
        """Gets list of neighbours of a given position.
        Args:
            coordinates (int, int): Coordinates to calculate neigbours for.
        Returns:
            :obj:`list` of (int,int): List of neighbour coordinates.
        """
        (cx, cy) = coordinates
        neighbors = []
        if cx - 1 >= 0:   neighbors.append((cx - 1, cy))
        if cx + 1 < self.board_size: neighbors.append((cx + 1, cy))
        if cx - 1 >= 0 and cy + 1 <= self.board_size - 1: neighbors.append((cx - 1, cy + 1))
        if cx + 1 < self.board_size and cy - 1 >= 0: neighbors.append((cx + 1, cy - 1))
        if cy + 1 < self.board_size: neighbors.append((cx, cy + 1))
        if cy - 1 >= 0:   neighbors.append((cx, cy - 1))
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
            visited (dict of (int,int):bool): Maps each set of coordinates to a bool: True if the hexagon was visited before.
        Returns:
            (bool): True if the current hexagon connects to a border hexagon.
        """
        if not self.is_color(move, color) or (move in visited and visited[move]): return False
        if self.border(color, move): return True
        visited[move] = True
        for n in self.get_neighbors(move):
            if self.traverse(color, n, visited): return True
        return False


    def get_move_list(self):
        """Gets the history of plyaed moves.
        Returns:
            :obj:`list` of (str): List of played moves.
        Examples:
            >>>HexBoard.get_move_list()
            ['a0', 'b1', 'c0']
        """
        return self.move_list


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


    def printBroad(self):
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
        turn_player = "blue" if self.blue_to_move else "red"
        if not self.game_over:
            print("It is currently " + turn_player + "\'s turn.")


    def create_GUI(self):
        """Binds buttons to the open window and calls Hexboard.draw_grid()
        """
        reset_button = tk.Button(self.window, text="Reset", command=self.reset_board)
        reset_button.pack()
        reset_button.update()
        reset_button.place(x=self.WIN_WIDTH - reset_button.winfo_width())

        undo_button = tk.Button(self.window, text="Undo", command=self.undo_move)
        undo_button.pack()
        undo_button.update()
        undo_button.place(x=reset_button.winfo_x() - reset_button.winfo_width(), y=reset_button.winfo_y())

        self.draw_grid()


    def on_click(self, event):
        """Determines action when the GUI is clicked."""
        if self.canvas.find_withtag(tk.CURRENT):
            if self.canvas.type(tk.CURRENT) == 'polygon' and not self.game_over:
                if self.canvas.itemcget(tk.CURRENT, 'fill') == 'white':
                    x, y = self.hex_to_coord(self.canvas.coords(tk.CURRENT))
                    if self.blue_to_move:
                        self.canvas.itemconfig(tk.CURRENT, fill="blue")
                        self.place((x, y), HexBoard.BLUE)
                    else:
                        self.canvas.itemconfig(tk.CURRENT, fill="red")
                        self.place((x, y), HexBoard.RED)
                    # print("You clicked on " + self.hex_to_string(self.canvas.coords(tk.CURRENT)))


    def reset_board(self):
        """Resets board and GUI"""
        if self.interactive_text:
            print("Reset the board.")
        for x in range(self.board_size):
            for y in range(self.board_size):
                self.board[x, y] = HexBoard.EMPTY
        self.move_list = []
        self.game_over = False
        self.blue_to_move = True

        if self.enable_GUI:
            for item in self.canvas.find_all():
                if self.canvas.type(item) == 'polygon':
                    self.canvas.itemconfig(item, fill="white")


    def undo_move(self):
        """Undoes the last made move"""
        if len(self.move_list) == 0:
            if self.interactive_text:
                print("No move to undo...")
            return
        else:
            if self.interactive_text:
                print("Undoing last move...")
            last_move = self.move_list.pop()
        x, y = self.string_to_coord(last_move)
        self.game_over = False
        if self.board[(x, y)] == HexBoard.RED:
            self.blue_to_move = False
        elif self.board[(x, y)] == HexBoard.BLUE:
            self.blue_to_move = True
        self.place((x, y), HexBoard.EMPTY)
        if self.enable_GUI:
            dx = 2
            dy = dx
            x_hex, y_hex = self.coord_to_hex((x, y))
            test = self.canvas.find_overlapping(x_hex - dx, y_hex - dy, x_hex + dx, y_hex + dy)
            for item in test:
                if self.canvas.type(item) == 'polygon':
                    self.canvas.itemconfig(item, fill="white")


    @staticmethod
    def hex_to_coord(coordinates):
        """Converts hexagon coordinates to grid coordinates.
        Args:
            coordinates (:obj:`list` of (double, double)): Hexagon vertex coordinates on the GUI window
        Returns:
            (int,int): Converted grid coordinates.
        """
        x_list = []
        y_list = []
        i = 0
        while i < len(coordinates):
            x_list.append(coordinates[i])
            y_list.append(coordinates[i + 1])
            i += 2
        x_average = sum(x_list) / len(x_list)
        y_average = sum(y_list) / len(y_list)
        y = round((y_average - HexBoard.Y_PADDING) / ((HexBoard.CROSS_LENGTH + HexBoard.SIDE_LENGTH) / 2))
        x_int = round((x_average - (HexBoard.X_PADDING + y * HexBoard.HEX_SIZE / 2)) / HexBoard.HEX_SIZE)
        return x_int, y


    @staticmethod
    def coord_to_hex(coordinates):
        """Converts grid coordinates to hexagon coordinates.
        Args:
            coordinates (int,int): Grid coordinates to convert.
        Returns:
            (double,double): Converted GUI coordinates of the hexagon's midpoint.
        """
        x = coordinates[0]
        y = coordinates[1]
        x_mid = HexBoard.X_PADDING + y * HexBoard.HEX_SIZE / 2 + x * HexBoard.HEX_SIZE
        y_mid = HexBoard.Y_PADDING + y * ((HexBoard.CROSS_LENGTH + HexBoard.SIDE_LENGTH) / 2)
        return x_mid, y_mid


    @staticmethod
    def hex_to_string(coordinates):
        """Converts hexagon coordinates to a string.
        Args:
            coordinates (:obj:`list` of (double, double)): Hexagon vertex coordinates on the GUI window
        Returns:
            str: Coordinates in string format
        """
        # x_list = []
        # y_list = []
        # i = 0
        # while i < len(coordinates):
        #     x_list.append(coordinates[i])
        #     y_list.append(coordinates[i + 1])
        #     i += 2
        # x_average = sum(x_list) / len(x_list)
        # y_average = sum(y_list) / len(y_list)
        # y = round((y_average - HexBoard.Y_PADDING) / ((HexBoard.CROSS_LENGTH + HexBoard.SIDE_LENGTH) / 2))
        # x_int = round((x_average - (HexBoard.X_PADDING + y * HexBoard.HEX_SIZE / 2)) / HexBoard.HEX_SIZE)
        # x = chr(x_int + 97)
        # return "" + str(x) + str(y)
        return HexBoard.coord_to_string(HexBoard.hex_to_coord(coordinates))


    def draw_grid(self):
        """Draws a grid of hexagons"""
        top_border = []
        bottom_border = []
        left_border = []
        right_border = []
        for yi in range(self.board_size):
            for i in range(self.board_size):
                xi = HexBoard.X_PADDING + yi * HexBoard.HEX_SIZE / 2 + i * HexBoard.HEX_SIZE
                y = HexBoard.Y_PADDING + yi * ((HexBoard.CROSS_LENGTH + HexBoard.SIDE_LENGTH) / 2)
                hexagon = RegularPolygon(6, HexBoard.HEX_SIZE, xi, y)
                self.canvas.create_polygon(hexagon.point_list, fill="white", outline="black")
                self.canvas.create_text(xi, y, text='' + str(chr(i + 97)) + str(yi))

                if yi == 0:
                    top_border.append(hexagon.point_list[-2])
                    top_border.append(hexagon.point_list[-1])
                    top_border.append(hexagon.point_list[0])
                    top_border.append(hexagon.point_list[1])
                    top_border.append(hexagon.point_list[2])
                    top_border.append(hexagon.point_list[3])
                elif yi == self.board_size - 1:
                    bottom_border.append(hexagon.point_list[-4])
                    bottom_border.append(hexagon.point_list[-3])
                    bottom_border.append(hexagon.point_list[6])
                    bottom_border.append(hexagon.point_list[7])
                    bottom_border.append(hexagon.point_list[4])
                    bottom_border.append(hexagon.point_list[5])
                if i == 0:
                    left_border.append(hexagon.point_list[-2])
                    left_border.append(hexagon.point_list[-1])
                    left_border.append(hexagon.point_list[-4])
                    left_border.append(hexagon.point_list[-3])
                    left_border.append(hexagon.point_list[6])
                    left_border.append(hexagon.point_list[7])
                elif i == self.board_size - 1:
                    right_border.append(hexagon.point_list[2])
                    right_border.append(hexagon.point_list[3])
                    right_border.append(hexagon.point_list[4])
                    right_border.append(hexagon.point_list[5])

        self.canvas.create_line(top_border, fill='red', width=HexBoard.BORDER_WIDTH)
        self.canvas.create_line(bottom_border, fill='red', width=HexBoard.BORDER_WIDTH)
        self.canvas.create_line(left_border, fill='blue', width=HexBoard.BORDER_WIDTH)
        self.canvas.create_line(right_border, fill='blue', width=HexBoard.BORDER_WIDTH)


def split_text(s):
    for k, g in groupby(s, str.isalpha):
        yield ''.join(g)

def valid_command(command, board_size):
    if bool(re.search(HexBoard.PATTERN, command)):
        if(command[0].isalpha() and ((ord(command[0]) - 97) < board_size) and command[1].isdigit() and int(command[1]) < board_size and int(command[1]) >= 0):
            return True
        else:
            return False
