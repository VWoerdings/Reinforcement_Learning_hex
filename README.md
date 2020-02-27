# Reinforcement_Learning_hex

Each section of the assignment has its own .py file.

## Search.py
This file contains a Hex implementation consisting of:
  A search function: alpha-beta
  A move generator
  A random evaluator function
It is possible to play against the random AI using either the text-based interface or the graphical interface.

## Eval.py
Similar to Search.py, but now the AI uses Dijkstra's shortest path algorithmm as a heuristic evaluation function.

## Experiment.py
This file uses the TrueSkill library to determine the skill rating of three AI, by playing them against each other.
This script rates the following AIs:
  AI with random evaluation
  AI with search depth 3 and Dijkstra evaluation
  AI with search depth 4 and Dijkstra evaluation

## IterativeDeepening_TranspositionTables.py
Similar to Experiment.py, but now the AIs have iterative deepening and transposition tables enabled.

# Files that are needed to run the previous scripts
## HexBoard.py
Contains a class that stores, computes and displays the hex board.
## RegularPolygon.py
Contains a class that calculates the vertices of a regular hexagon. Needed to draw the graphical interface.
## TerminatorHex.py
Contains a class that is responsible for AI behaviour. It contains several heuristic evaluators and move generators.
