# train_hex.py
This script will train AlphaZero to play Hex
Here you can edit parameters for the MCTS and iterations, epochs, etc.

# hex
This package contains the implementation of the neural network and Hex board and logic.

# Coach.py
This class executes the training loops.

# Game.py
A class template for a game implementation. Implemented for Hex in the hex package.

# MCTS.py
Implementation of MCTS for AlphaZero.
This file contains a temporary fix. Sometimes <em>counts</em> in <em>getActionProb()</em> will be 0. As a result, when <em>temp</em> is 0, the algorithms will pick a rondom move. Sometimes, this move is already played.
For the temporary fix, when <em>counts</em> is 0, we recalculate possible moves and randomly select one of those. This fix should not be necessary.

# NeuralNet.py
A class template for a neural network. Implemented for Hex in the hex package.

# pit.py
Allows you to play against a trained AI (not ready yet for hex). Uses Arena.py

# utils.py
Some utils, needs to be inclueded.