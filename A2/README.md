# Reinforcement_Learning_hex Assignment 2

Each section of the assignment has its own .py file.

## Experiment.py
Calculates the rating of several MCTS algorithms by playing it against an ID_TT algorithm

## Tune.py
Calculates the optimal parameters for the MCTS algorithm

## FramedEvaluation.py
This script evaluates positional advantage according to a heuristic evaluation function (see TerminatorHex)
for N_EPOCHS games with two AIs. This allows one to see strengths and weaknesses of AIs measured over the
span of games.

## HexBoard.py
This script contains the hex board that stores and computes the state of a hex board.

## MCTSHex.py
This script contains MCTS game AI implementation and MCTS node

# Files that are needed to run the previous scripts
## HexBoard.py
Contains a class that stores, computes and displays the hex board.

## RegularPolygon.py
Contains a class that calculates the vertices of a regular hexagon. Needed to draw the graphical interface.
