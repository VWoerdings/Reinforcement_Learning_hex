import gym
import random
import time
import numpy as np
import os

from BreakoutPlay import *
from BreakoutBuffers import *

# Breakout DQN Learner using the trajectory type buffer
# Main loop:       
BUFFER_SIZE = 6 # 10 games
CYCLES_FOR_TRANSFER = 5
N_GAMES_PER_PLAY_CYCLE = 1 # new games per master epoch
GAME_ACTIONS_LIMIT = 10000 # maximum actions per game (prevent getting stuck)
N_SAMPLES_PER_LEARN_CYCLE = 40
N_EPOCHS_PER_LEARN_CYCLE = 15
N_CYCLES_PERFORMANCE_EVAL = 0
N_EPOCHS_MASTER = 30
EPSILON = 0.8
DISCOUNT = 0.98
FRAME_RATE = 0.02
DISABLE_RENDERING = False # whether to disable rendering the game
EXPERIENCE_BUFFER_MODE = 'trajectory' # experience buffer type: 'simple', 'posisplit' or 'trajectory'

WEIGHT_LOAD_PATH = None # if none, do not load weights to DQNs, initialise randomly
STORE_WEIGHTS = True # whether to store the DQN weights after completeing the run (stores target network last values)
WEIGHT_STORE_PATH = os.getcwd() + "/weights"
WEIGHT_STORE_NAMESTAMP = "latest" # if None: generate a time-based namestamp; if some string: can overwrite that file!

#np.random.seed(333)
#random.seed(333)
GAME_SEED = None # environment seed

learner = BreakoutDQNLearner(BUFFER_SIZE, CYCLES_FOR_TRANSFER, DISCOUNT,
                             load_weights=WEIGHT_LOAD_PATH, game_seed=GAME_SEED, buffer_mode=EXPERIENCE_BUFFER_MODE)
print(">__main__: Filling buffer (games:", BUFFER_SIZE, "total)")
for i in range(BUFFER_SIZE): # buffer filling
    print("Filling buffer: game", i + 1)
    count = 0
    while True: # store a full game
        count += 1
        tup = learner.takeActionAndStoreExperience(epsilon=EPSILON, strategy='random')
        if tup[learner.buffer_indices['game_over']] == True or count > GAME_ACTIONS_LIMIT:
            if count > GAME_ACTIONS_LIMIT:
                print(">__main__: game buffer filler reached action limit, aborting fill")
            break

for i in range(N_EPOCHS_MASTER):
    print("Master epoch", i + 1)
    for _ in range(N_GAMES_PER_PLAY_CYCLE):
        count = 0
        while True: # store a full game
            count += 1
            tup = learner.takeActionAndStoreExperience(epsilon=EPSILON, strategy='random')
            if tup[learner.buffer_indices['game_over']] == True or count > GAME_ACTIONS_LIMIT:
                if count > GAME_ACTIONS_LIMIT:
                    print(">__main__: game buffer filler reached action limit, aborting fill")
                break
    learner.updateNetwork(nsamples_replay_buffer=N_SAMPLES_PER_LEARN_CYCLE, epochs=N_EPOCHS_PER_LEARN_CYCLE)
    total_score = 0
    state = learner.game.clone_full_state()
    for _ in range(N_CYCLES_PERFORMANCE_EVAL):
        learner.resetAndRandomNonZeroMove()
        tup = learner.takeActionAndStoreExperience(epsilon=0.95, do_not_store=True)
        total_score += tup[learner.buffer_indices['reward']]
    learner.game.restore_full_state(state)
    print("Total score for master epoch:", total_score)

if STORE_WEIGHTS:
    print(">__main__: storing weights")
    if not os.path.isdir(WEIGHT_STORE_PATH):
        print(">__main__: creating directory:", WEIGHT_STORE_PATH)
        os.mkdir(WEIGHT_STORE_PATH)
    timest = time.localtime(time.time())
    if WEIGHT_STORE_NAMESTAMP == None:
        namestamp = "breakout_weights_" + str(timest.tm_mon) + str(timest.tm_mday) + str(timest.tm_hour) + str(timest.tm_min) + str(timest.tm_sec)
    else:
        namestamp = WEIGHT_STORE_NAMESTAMP
    learner.target_network.model.save_weights((WEIGHT_STORE_PATH + "/" + namestamp))

# Test the AI in NUM_GAMES games
NUM_GAMES = 5
learner.resetAndRandomNonZeroMove()
total_score = 0
game_score = 0
games_completed = 0
complete = False
while not complete:
    tup = learner.takeActionAndStoreExperience(epsilon=1.00, do_not_store=True)
    #print("Action", tup[learner.buffer_indices['action']])
    learner.render(FRAME_RATE, disable=DISABLE_RENDERING)
    total_score += tup[learner.buffer_indices['reward']]
    game_score += tup[learner.buffer_indices['reward']]
    if tup[learner.buffer_indices['game_over']] == True:
        print("Game finished; score:", game_score)
        game_score = 0
        games_completed += 1
        if games_completed >= NUM_GAMES:
            complete = True
        else:
            learner.resetAndRandomNonZeroMove()
print("Average score", str((total_score / NUM_GAMES)))
