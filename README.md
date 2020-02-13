# Reinforcement_Learning_hex

HexGame.py is the main program.

HexBoard.py contains a class that stores, computes and displays the hex board.

RegularPolygon.py contains a class that calculates the vertices of each hexagon.

class HexBoard(builtins.object)
 |  HexBoard(board_size, enable_GUI=False, interactive_text=False)
 |  
 |  A class that stores and computes the state of a hex board.
 |  
 |  Methods defined here:
 |  
 |  __init__(self, board_size, enable_GUI=False, interactive_text=False)
 |      Initializes the board and GUI if applicable.
 |      Args:
 |          board_size (int): Size of the hexagon grid.
 |          enable_GUI (bool, optional): Enables an interactive GUI. Default is False.
 |          interactive_text (bool, optional): Enables an interactive text mode. Sets enable_GUI to False. Default is False. Maximum board size for this mode is currently 10.
 |  
 |  border(self, color, move)
 |      Checks if a hexagon is located on the right (if blue) or bottom (if blue) border of the board.
 |      Args:
 |          color: (int) Color of the hexagon.
 |          move: (int,int) Coordinates of the hexagon.
 |      Returns:
 |          (bool): True if the hexagon is located at the border of the board.
 |  
 |  check_and_print_winner(self)
 |      Checks if there is a winner and prints to the GUI or console if applicable
 |      Returns:
 |          (bool): True if a winner was found, False otherwise.
 |  
 |  check_win(self, color)
 |      Checks if there is a winner.
 |      Returns:
 |          (bool): True if there is a winner, false otherwise.
 |  
 |  create_GUI(self)
 |      Binds buttons to the open window and calls Hexboard.draw_grid()
 |  
 |  draw_grid(self)
 |      Draws a grid of hexagons
 |  
 |  get_color(self, coordinates)
 |      Finds the color at given coordinates.
 |      Args:
 |          coordinates (int, int): X and y coordinates to check.
 |      Returns:
 |          int: Color at the given position.
 |  
 |  get_move_list(self)
 |      Gets the history of plyaed moves.
 |      Returns:
 |          :obj:`list` of (str): List of played moves.
 |      Examples:
 |          >>>HexBoard.get_move_list()
 |          ['a0', 'b1', 'c0']
 |  
 |  get_neighbors(self, coordinates)
 |      Gets list of neighbours of a given position.
 |      Args:
 |          coordinates (int, int): Coordinates to calculate neigbours for.
 |      Returns:
 |          :obj:`list` of (int,int): List of neighbour coordinates.
 |  
 |  interactive_text_loop(self)
 |      Contains an infinite loop that reads and handles commands from the console.
 |  
 |  is_color(self, coordinates, color)
 |      Checks if the board is a given color at given coordinates.
 |      Args:
 |          coordinates (int, int): X and y coordinates to check.
 |          color (int): Color to compare to.
 |      Returns:
 |          bool: True if the given position is equal to the given color, False otherwise.
 |  
 |  is_empty(self, coordinates)
 |      Checks if the board is empty at given coordinates.
 |      Args:
 |          coordinates (int, int): X and y coordinates to check.
 |      Returns:
 |          bool: True if the given position is empty, False otherwise.
 |  
 |  is_game_over(self)
 |      Checks if game is over.
 |      Returns:
 |          bool: True if the game is over, False otherwise.
 |  
 |  is_valid(self, coordinates)
 |      Checks if coordinates are within the bounds of the grid.
 |      Args:
 |          coordinates (int,int): Coordinates to check.
 |      Returns: True if coordinates are on the board, False otherwise
 |  
 |  on_click(self, event)
 |      Determines action when the GUI is clicked.
 |  
 |  place(self, coordinates, color)
 |      Places a given color at given coordinates.
 |      Args:
 |          coordinates (int, int): X and y coordinates to check.
 |          color (int): Color to place.
 |  
 |  print(self)
 |      Prints the board to the console.
 |  
 |  reset_board(self)
 |      Resets board and GUI
 |  
 |  traverse(self, color, move, visited)
 |      Traverses hexagons from top to bottom (if red) or from left to right (if blue).
 |      Args:
 |          color (int): Color of the current hexagon.
 |          move (int,int): Coordinates of the current hexagon.
 |          visited (dict of (int,int):bool): Maps each set of coordinates to a bool: True if the hexagon was visited before.
 |      Returns:
 |          (bool): True if the current hexagon connects to a border hexagon.
 |  
 |  undo_move(self)
 |      Undoes the last made move
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  coord_to_hex(coordinates)
 |      Converts grid coordinates to hexagon coordinates.
 |      Args:
 |          coordinates (int,int): Grid coordinates to convert.
 |      Returns:
 |          (double,double): Converted GUI coordinates of the hexagon's midpoint.
 |  
 |  coord_to_string(coordinates)
 |      Converts grid coordinates to a string.
 |      Args:
 |          coordinates (int, int): Grid coordinates to convert.
 |      Returns:
 |          str: Coordinates in string format
 |      Examples:
 |          >>>HexBoard.coord_to_string((0,0))
 |          'a0'
 |  
 |  get_opposite_color(current_color)
 |      Gets opposite to given color.
 |      Args:
 |          current_color (int): Input color.
 |      Returns:
 |          (int): Opposite color.
 |  
 |  hex_to_coord(coordinates)
 |      Converts hexagon coordinates to grid coordinates.
 |      Args:
 |          coordinates (:obj:`list` of (double, double)): Hexagon vertex coordinates on the GUI window
 |      Returns:
 |          (int,int): Converted grid coordinates.
 |  
 |  hex_to_string(coordinates)
 |      Converts hexagon coordinates to a string.
 |      Args:
 |          coordinates (:obj:`list` of (double, double)): Hexagon vertex coordinates on the GUI window
 |      Returns:
 |          str: Coordinates in string format
 |  
 |  print_command_help()
 |      Prints a list of available commands for the interactive text mode
 |  
 |  string_to_coord(input_string)
 |      Converts string to grid coordinates.
 |      Args:
 |          input_string (str): String to convert.
 |      Returns:
 |          (int,int): Converted grid coordinates.
 |      Examples:
 |          >>>HexBoard.string_to_coord('a0')
 |          (0,0)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  APOTHEM = 25.0
 |  
 |  BLUE = 1
 |  
 |  BORDER_WIDTH = 3
 |  
 |  CROSS_LENGTH = 57.735026918962575
 |  
 |  EMPTY = 3
 |  
 |  HEX_SIZE = 50
 |  
 |  MIN_PADDING = 25
 |  
 |  MIN_PADDING_X = 50.0
 |  
 |  MIN_PADDING_y = 53.86751345948129
 |  
 |  RED = 2
 |  
 |  SIDE_LENGTH = 28.867513459481287
 |  
 |  X_PADDING = 50.0
 |  
 |  Y_PADDING = 53.86751345948129
