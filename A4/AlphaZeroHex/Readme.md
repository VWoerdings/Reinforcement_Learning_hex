# train_hex.py
This script will train AlphaZero to play Hex
Here you can edit parameters for the MCTS and iterations, epochs, etc.

# Tournament.py
This script plays a tournament between an ID-TT, a MCTS and an AlphaZero player and determines their rating.

# Tournament_old.py
This script plays a tournament between two AlphaZero players.

# Hyperparameters.py
This script plays a tournament between three AlphaZero players with varying exploration and determines their rating (similar to Tournament.py).

# hex/
This package contains the implementation of the neural network, Hex board and logic.

# old_ai/
This package contains previous versions of the ID-TT and MCTS AIs.

# trained_networks/
This folder contains trained AlphaZero networks that can be loaded.

# Arena.py
This class plays two AlphaZero agents against eachother. This is used to train the networks.

# Coach.py
This class executes the training loops.

# Game.py
A class template for a game implementation. Implemented for Hex in the hex package.

# MCTS.py
Implementation of MCTS for AlphaZero.

# NeuralNet.py
A class template for a neural network. Implemented for Hex in the hex package.

# utils.py
Some utils, needs to be inclueded.
