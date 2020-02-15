
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

    def terminateMinMax(self , depth):
        if(depth == 0 or self.board.gameOver):
            return