import numpy as np
import copy
import random

# A simple game to test Deep Q-Networks
# Move a dot in 1 dimension (left or right)
# At every step: receive a reward according to some positional reward function
# So: discrete action space, continuous state space
# Game ends when passing some position threshold or when max cycles exceeded
# Author: Ruben Walen

class DiscreteActionSpace:
    def __init__(self, discrete_size):
        self.n = discrete_size

    def sample(self):
        return random.randrange(self.n)

class SimpleGame:
    def __init__(self):
        self.max_cycles = 1000
        self.frame_y_size = 40
        self.min_x = -20
        self.max_x = 20
        self.kill_x = 8
        self.accel = 1
        self.dt = 0.1
        self.slip = 0.9
        self.pos_reward = lambda x: -1 * (x + 6) #lambda x : (x - 3)**2
        self.action_space = DiscreteActionSpace(3)
        self.randseed = None
        self.reset()

    def get_action_meanings(self):
        return ["NONE", "LEFT", "RIGHT"]
    
    def step(self, action):
        if action not in range(self.action_space.n):
            raise Exception("@SimpleGame.step: action index out of discrete range")

        if self.game_over:
            print(">SmallGame.step: game over, resetting")
            self.reset()

        self.v = self.v * self.slip
        if action == 1:
             self.v -= self.accel * self.dt
        elif action == 2:
            self.v += self.accel * self.dt

        self.change_position()
        reward = self.get_reward()
        frame_start = copy.copy(self.frame) # TODO: return?
        frame_end = self.change_frame()
        self.n_cycles += 1
        if self.x > self.kill_x or self.n_cycles > self.max_cycles:
            self.game_over = True
        return (frame_end, reward, self.game_over, None)
        
    def change_position(self):
        self.x += self.v * self.dt
        return self.x
    
    def get_reward(self):
        return self.pos_reward(self.x)

    def change_frame(self):
        self.frame = np.zeros((int(self.max_x - self.min_x + 1), self.frame_y_size, 1)).astype('int32')
        self.frame[:, 10, 0] = 1 # clutter
        self.frame[int(self.x - self.min_x), 3, 0] = 1  
        return self.frame

    def reset(self):
        self.n_cycles = 0
        self.x = 0
        self.v = 0
        self.frame = np.zeros((int(self.max_x - self.min_x + 1), self.frame_y_size, 1)).astype('int32')
        self.frame[int(self.x - self.min_x), 3, 0] = 1
        self.frame[:, 10, 0] = 1 # clutter
        self.game_over = False
        if self.randseed is not None:
            random.seed(self.randseed)
        return

    def render(self):
        array = self.frame
        for i in range(array.shape[0]):
            string = ""
            for j in range(array.shape[1]):
                string += str(array[i, j, 0])
            print(string)
        return

    def seed(self, seed):
        self.randseed = seed
        return

if __name__ == '__main__':
##    newgame = SimpleGame()
##    newgame.render()
##    for _ in range(3):
##        tup = newgame.step(1)
##        print("---------")
##        #newgame.render()
##        print(">REWARD", tup[1])

    from BreakoutPlay import *

    #bknl = BreakoutDQNLearner(20, 5, 1, custom_game=SimpleGame(), convolutional_layers=[(10, 5, (1, 1)), (20, 5, (1, 1))])

    BUFFER_SIZE = 500 # size of the replay buffer
    CYCLES_FOR_TRANSFER = 6 # cycles to wait before transferring prediction weights to target network
    N_ACTIONS_PER_PLAY_CYCLE = 25 # number of actions to sample in each master epoch
    N_SAMPLES_PER_LEARN_CYCLE = 40 # number of samples to train with in master epoch training step
    N_EPOCHS_PER_LEARN_CYCLE = 10 # number of epochs to train per master epoch
    N_CYCLES_PERFORMANCE_EVAL = 0 # number of cycles for performance evaluation during each master epoch (slows down the algorithm)
    N_EPOCHS_MASTER = 1000
    EPSILON = 0.8 # epsilon-greedy exploration parameter (during training)
    DISCOUNT = 0.80 # discount factor during training
    EMBELLISH_REWARD_FACTOR = 10 # linear reward scaling
    FRAME_RATE = 0.02 # frame rate for rendering steps
    EXPERIENCE_BUFFER_MODE = 'posisplit' # experience buffer type: 'simple', 'posisplit' or 'trajectory'
    DISABLE_RENDERING = True

    #np.random.seed(333)
    #random.seed(333)
    GAME_SEED = None # environment seed
    
    learner = BreakoutDQNLearner(BUFFER_SIZE, CYCLES_FOR_TRANSFER, DISCOUNT,
                                 load_weights=None, game_seed=GAME_SEED, buffer_mode=EXPERIENCE_BUFFER_MODE,
                                 embellish_reward_factor=EMBELLISH_REWARD_FACTOR, custom_game=SimpleGame())
    print(">__main__: Filling buffer (samples:", BUFFER_SIZE, "total)")
    for i in range(BUFFER_SIZE): # buffer filling
        #print("Filling buffer: cycle", i + 1)
        learner.takeActionAndStoreExperience(epsilon=EPSILON, strategy='random')
    #learner.render(FRAME_RATE, disable=DISABLE_RENDERING)
    for i in range(N_EPOCHS_MASTER):
        print("Master epoch", i + 1)
        for _ in range(N_ACTIONS_PER_PLAY_CYCLE):
            learner.takeActionAndStoreExperience(epsilon=EPSILON)
            #learner.render(FRAME_RATE, disable=DISABLE_RENDERING)
        learner.updateNetwork(nsamples_replay_buffer=N_SAMPLES_PER_LEARN_CYCLE, epochs=N_EPOCHS_PER_LEARN_CYCLE)
        total_score = 0
        #state = learner.game.clone_full_state() # TODO: implement
        for _ in range(N_CYCLES_PERFORMANCE_EVAL):
            learner.resetAndRandomNonZeroMove()
            tup = learner.takeActionAndStoreExperience(epsilon=0.95, do_not_store=True)
            total_score += tup[learner.buffer_indices['reward']]
        #learner.game.restore_full_state(state)
        print("Total score for master epoch:", total_score)

    # Test the AI in NUM_GAMES games
    NUM_GAMES = 5
    learner.resetAndRandomNonZeroMove()
    total_score = 0
    game_score = 0
    games_completed = 0
    max_Q_vector = []
    actions_taken = []
    complete = False
    while not complete:
        tup = learner.takeActionAndStoreExperience(epsilon=0.98, do_not_store=True)
        if games_completed == 0: # first game only
            Q_vector = learner.prediction_network.predictQVectorFromFrame(tup[learner.buffer_indices['start_frame']])
            max_Q = np.max(Q_vector)
            max_Q_vector.append(max_Q)
            actions_taken.append(tup[learner.buffer_indices['action']])
        print("Action", tup[learner.buffer_indices['action']])
        print("Position", learner.game.x)
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
