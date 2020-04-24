import tensorflow as tf
from tensorflow.keras import layers, models
import gym
import random
import time
import numpy as np
import os
import matplotlib.pyplot as plt

from BreakoutBuffers import *

# RL Assigment 3: DQN Learning; Part 2: Atari Breakout
# April 2020
# Abhishek Sira Chandrashekar, Virgil Woerdings, Ruben Walen
# BreakoutPlay: contains the neural code, learning code and main loop
#
# TODO:
# discount factor DONE
# kernel size, stride checking DONE
# check parameters
# ...
#
# ENHANCEMENTS:
# trajectories buffer: instead of storing individual samples...
#... store whole-game trajectories. Backpropagate (discounted) rewards...
#... along the game trajectory when a positive reward is received
# main/positive buffer split: always attempt to train network with a certain...
#... proportion of positive reward samples kept separately in the buffer

class BreakoutNetwork:
    def __init__(self, frame_size, resize_factor, n_actions, loss_function, optimiser, load_weights=None):
        # The DQN Network implementation (static architecture, see below).
        # The architecture has a variable number of convolutional layers, which are then collected using a dense layer, which outputs to a dense action space layer.
        # Params:
        # frame_size (list/array/tuple of int): the size of the frame
        # resize_factor (float): automatic scaling factor for the frame
        # n_actions (int): the size of the discrete action space
        # loss_function (tf loss function): a TensorFlow loss function for training
        # optimiser (tf optimizer): a TensorFlow network optimizer
        # load_weights (string): a filepath to a network weights file
        self.original_frame_size = frame_size
        self.resize_factor = resize_factor
        self.reduced_frame_size = np.array([int(self.original_frame_size[0] * resize_factor),
                                            int(self.original_frame_size[1] * resize_factor),
                                            int(self.original_frame_size[2])])
        self.n_actions = n_actions

        # construct the network
        self.network_params = [(32, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (64, (3, 3), (1, 1))] # convolutional feature dimensionality (output) and stride: every entry = 1 conv. layer
        # [nfilters, (kernel_size_X, kernel_size_Y), (stride_X, stride_Y)]
        # previous tries: [32, (3, 3), (1, 1)]
        # [(32, (3, 3), (1, 1)), (64, (3, 3), (1, 1)), (64, (3, 3), (1, 1))]
        self.initialiser = tf.keras.initializers.RandomUniform(minval=-0.0002, maxval=0.0005, seed=None) # a weights initialiser: this prevents high initial Q-values

        model = models.Sequential()
        for i, entry in enumerate(self.network_params):
            model.add(layers.Conv2D(entry[0], entry[1], strides=entry[2], activation='linear',
                                    input_shape=self.reduced_frame_size, kernel_initializer=self.initialiser)) # convolutional layer
            model.add(layers.MaxPooling2D((2, 2))) # max pooling to lower image size
        
        model.add(layers.Flatten()) # flatten feature maps
        model.add(layers.Dense(self.network_params[-1][0],
                               activation='relu', kernel_initializer=self.initialiser)) # first dense layer: connect to last conv. layer
        model.add(layers.Dense(n_actions, activation='linear', kernel_initializer=self.initialiser)) # second dense layer: output actions

        self.model = model
        self.model.compile(loss=loss_function, optimizer=optimiser)

        if load_weights != None: # load weights from a weights filepath
            self.model.load_weights(load_weights)

    def predictQVectorFromFrame(self, frames):
        # Predict the Q-values of frames. Automatically resizes images.
        if len(frames.shape) == 3:
            frames = np.reshape(frames, (1, frames.shape[0], frames.shape[1], frames.shape[2])) # single-image batch
        resized_images = tf.image.resize(frames, self.reduced_frame_size[0:2]) # resize images first
        return self.model.predict(resized_images)
        #values = frame
        #for layer in self.model._layers:
        #    values = layer(values) # propagate through the layer
        #return values # return the values in the output layer after propagating through all layers

    def fit(self, input_frames, output_matrix, batch_size, epochs):
        # TensorFlow fitting, using output_matrix as targets and input_frames as inputs. This is called in DQNLeaner.updateNetwork()
        resized_images = tf.image.resize(input_frames, self.reduced_frame_size[0:2]) # resize images first
        self.model.fit(resized_images, output_matrix, batch_size=batch_size, epochs=epochs)
        return

class BreakoutDQNLearner:
    def __init__(self, buffer_size, cycles_per_network_transfer, discount_factor, load_weights=None, game_seed=None, buffer_mode='simple', embellish_reward_factor=1):
        # Our implementation of a DQN-based Q-learning algorithm. This class handles experience storing, game stepping, network updates, action strategy, and more.
        # Params:
        # buffer_size (int): the maximum size of the replay buffer
        # cycles_per_network_transfer (int): how many cycles before the prediction network weights are transferred to the target network
        # discount_factor (float, 0 to 1): the discount factor for the policy learner
        # load_weights (string): a filename to load network weights from, for the load/save system
        # game_seed (None or int): a seed for the Atari environment, if None: the seed is random each game
        # buffer_mode ('simple' OR 'posisplit' OR 'trajectory'): the type of buffer used, seed BreakoutBuffers.py
        # embellish_reward_factor (float): a linear scaling factor for rewards sampled from actions. May make training the networks easier?
        
        self.buffer_mode = buffer_mode
        if buffer_mode == 'simple':
            self.buffer = BreakoutExperienceBuffer(buffer_size)
        elif buffer_mode == 'posisplit':
            self.buffer = BreakoutExperiencePosisplitBuffer(buffer_size, buffer_size * 0.2)
        elif buffer_mode == 'trajectory':
            self.buffer = BreakoutExperienceTrajectoryBuffer(buffer_size, auto_backpropagation_discount=discount_factor) # ATTN: buffer_size is now the size in games, not in samples!
        else:
            raise Exception(("@BreakoutDQNLearner.__init__: invalid buffer mode: " + buffer_mode))
            
        self.n_updates_count = 0 # how many times the network(s) was updated
        self.cycles_per_network_transfer = cycles_per_network_transfer # after how many update cylces we update the target network...
        self.discount_factor = discount_factor # the Q-policy learner discount factor
        self.embellish_reward_factor = embellish_reward_factor # this is a linear scaling factor for the rewards
        #... with the prediction network weights

        self.game = gym.make('Breakout-v0')
        self.game_seed = game_seed
        self.game.seed(game_seed)
        self.current_frame = self.game.reset()
        self.current_frame, _, self.game_over, _ = self.game.step(random.choice(range(1, self.game.action_space.n)))
        self.game_over = False
        self.last_frame_time = 0
        self.action_space_size = self.game.action_space.n

        # target and prediction networks separated to reduce target instability (double DQN)
        #self.opt = tf.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # stochastic gradient descent
        #self.opt = tf.optimizers.RMSprop(learning_rate=0.001, rho=0.9) # RMSprop
        self.opt = tf.optimizers.Adagrad(learning_rate=0.01) # adagrad
        self.resize_factor = 0.7
        self.target_network = BreakoutNetwork(self.current_frame.shape, self.resize_factor, self.action_space_size, "mean_squared_error", self.opt, load_weights=load_weights)
        self.prediction_network = BreakoutNetwork(self.current_frame.shape, self.resize_factor, self.action_space_size, "mean_squared_error", self.opt, load_weights=load_weights)

        self.buffer_indices = {'start_frame': 0, 'action': 1, 'reward': 2, 'game_over': 3, 'result_frame': 4}

        print(">BreakoutDQNLearner: Q check (initial frame)")
        print(self.target_network.predictQVectorFromFrame(self.current_frame))
        print("Actions:", self.game.get_action_meanings())

    def getMostPrudentAction(self, strategy='epsilon-greedy', **kwargs):
        # Use a strategy function to determine the most prudent action given the current frame (state) and environment
        # strategy ('random' OR 'epsilon-greedy'): the prudence strategy for picking the action. May require one or more kwargs.
        Q_vector = self.prediction_network.predictQVectorFromFrame(self.current_frame)
        if strategy == 'epsilon-greedy':
            epsilon = kwargs['epsilon']
            action = self._selectEpsilonGreedy(Q_vector, epsilon=epsilon)
        elif strategy == 'random':
            action = self._selectRandom(Q_vector)
        else:
            raise Exception("@BreakoutDQNLearner.getMostPrudentAction: unknown strategy")
        return action

    def takeActionAndStoreExperience(self, do_not_store=False, strategy='epsilon-greedy', **kwargs):
        # Take an action according to the action strategy (see getMostPrudentAction()) and store the experience in the buffer
        # (unless do_not_store is True)
        action = self.getMostPrudentAction(strategy, **kwargs)
        frame, reward, game_over, _ = self.game.step(action)

        reward = reward * self.embellish_reward_factor # we can embellish rewards linearly in this way
        tup = (self.current_frame, action, reward, game_over, frame)
        if not do_not_store:
            self.buffer.put(tup) # put new action tuple in the experience replay buffer
        self.current_frame = frame
        self.game_over = game_over
        if self.game_over:
            self.resetAndRandomNonZeroMove()
        return tup

    def updateNetwork(self, use_replay_buffer=True, nsamples_replay_buffer=1, train_batch_size='auto', epochs=1, experiences=None):
        # One cycle of DQN network updating. Trains the prediction network, and may then transfer to the target network (see self.cycles_per_network_transfer))
        # The core training loop uses a root-mean-squared error loss function that should be 0 for every action node in the DQN that is not the
        #   action node corresponding to the current sampled experience's action. The loss for that action node follows the Q-policy learning loss
        # Params:
        # use_replay_buffer (bool): whether to use the replay buffer - if False you must provide experiences yourself
        # nsamples_replay_buffer (int): how many samples to draft from the replay buffer
        # train_batch_size ('auto' or int): if not 'auto', you can set the training batch size to something not equal to the number of samples
        # epochs (int): number of epochs to train
        # experiences (None or list of experiences): a list of experiences if use_replay_buffer == False
        if train_batch_size == 'auto':
            train_batch_size = nsamples_replay_buffer
        if use_replay_buffer:
            experience_batch = self.buffer.sample(nsamples_replay_buffer) # throws exception: buffer content too small
        else:
            experience_batch = experiences

        # the core training loop of the algorithm starts here
        target_matrix = np.zeros((nsamples_replay_buffer, self.action_space_size)) # matrix of target Q values
        input_frames = [None for _ in range(nsamples_replay_buffer)] # the frames to use as inputs
        for i, exp in enumerate(experience_batch):
            target_matrix[i, :] = self.prediction_network.predictQVectorFromFrame(exp[self.buffer_indices['start_frame']]) # no loss for non-represented action
            if exp[self.buffer_indices['game_over']] == False: # not a game over state - add the max Q of the next frame to the action taken
                max_Q = np.max(self.target_network.predictQVectorFromFrame(exp[self.buffer_indices['result_frame']])) # from frame resulting from action
                target_matrix[i, exp[self.buffer_indices['action']]] = self.discount_factor * max_Q # replace by max_Q target
            target_matrix[i, exp[self.buffer_indices['action']]] += exp[self.buffer_indices['reward']] # update with reward associated with that action in that sample
            input_frames[i] = exp[self.buffer_indices['start_frame']]

        self.prediction_network.fit(np.array(input_frames), target_matrix, train_batch_size, epochs)
        self.n_updates_count += 1
        if self.n_updates_count % self.cycles_per_network_transfer == 0: # transfer prediction network to train network
            self.target_network.model.set_weights(self.prediction_network.model.get_weights())
        return

    def render(self, frame_rate_mills, disable=False):
        # Wait for a maximum of frame_rate_mills milliseconds per render cycle, draw the game screen with most recent action
        # disable (bool): useful for command-line runs
        if not disable:
            time_now = time.time()
            frame_wait = max(0, (frame_rate_mills - (time_now - self.last_frame_time)))
            time.sleep(frame_wait)
            self.last_frame_time = time_now
            self.game.render()
        return
    
    def _selectEpsilonGreedy(self, Q_vector, epsilon=0.8):
        # Use epsilon-greedy strategy to select an action
        # Q_vector contains the predicted Q-value (value function) for each action
        # Returns an integer corresponding to the chosen action
        if random.random() < epsilon: # pick highest rewarding (exploit)
            max_values = np.where(Q_vector == np.amax(Q_vector))[0] # all maximum value indices; prevent determinism on first index (argmax)
            chosen_index = random.choice(max_values) # pick one of the max values at random
            return chosen_index
        else: # pick random for exploration
            return random.randrange(Q_vector.shape[1])
        pass

    def _selectRandom(self, Q_vector):
        # Select a random action
        # Q_vector contains the predicted Q-value (value function) for each action
        # Returns an integer corresponding to the chosen action
        return random.randrange(Q_vector.shape[1])

    def resetAndRandomNonZeroMove(self):
        # Reset the game, reseed and then do a random move to start the game again
        self.game.reset() # reset the game completely
        self.game.seed(self.game_seed) # reseed
        self.current_frame, _, self.game_over, _ = self.game.step(random.choice(range(1, self.game.action_space.n))) # prevent getting stuck in zero-moves
        return

# Main loop:       
if __name__ == "__main__":
    # DQN Learning on Atari Breakout
    BUFFER_SIZE = 500 # size of the replay buffer
    CYCLES_FOR_TRANSFER = 10 # cycles to wait before transferring prediction weights to target network
    N_ACTIONS_PER_PLAY_CYCLE = 10 # number of actions to sample in each master epoch
    N_SAMPLES_PER_LEARN_CYCLE = 25 # number of samples to train with in master epoch training step
    N_EPOCHS_PER_LEARN_CYCLE = 10 # number of epochs to train per master epoch
    N_CYCLES_PERFORMANCE_EVAL = 0 # number of cycles for performance evaluation during each master epoch (slows down the algorithm)
    N_EPOCHS_MASTER = 10
    EPSILON = 0.8 # epsilon-greedy exploration parameter (during training)
    DISCOUNT = 0.95 # discount factor during training
    EMBELLISH_REWARD_FACTOR = 10 # linear reward scaling
    FRAME_RATE = 0.02 # frame rate for rendering steps
    DISABLE_RENDERING = False # whether to disable rendering the game
    DISABLE_PLOTTING = False # disable some plot making (see end of this file)
    EXPERIENCE_BUFFER_MODE = 'posisplit' # experience buffer type: 'simple', 'posisplit' or 'trajectory'

    WEIGHT_LOAD_PATH = None # if none, do not load weights to DQNs, initialise randomly
    STORE_WEIGHTS = True # whether to store the DQN weights after completeing the run (stores target network last values)
    WEIGHT_STORE_PATH = os.getcwd() + "/weights"
    WEIGHT_STORE_NAMESTAMP = "latest" # if None: generate a time-based namestamp; if some string: can overwrite that file!

    #np.random.seed(333)
    #random.seed(333)
    GAME_SEED = None # environment seed
    
    learner = BreakoutDQNLearner(BUFFER_SIZE, CYCLES_FOR_TRANSFER, DISCOUNT,
                                 load_weights=WEIGHT_LOAD_PATH, game_seed=GAME_SEED, buffer_mode=EXPERIENCE_BUFFER_MODE,
                                 embellish_reward_factor=EMBELLISH_REWARD_FACTOR)
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

    if not DISABLE_PLOTTING:
        fig, ax1 = plt.subplots()
        ax1.plot(max_Q_vector)
        ax1.set_title("Max Q-values per action frame (one game)")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Q-value")

        fig, ax2 = plt.subplots()
        ax2.hist(actions_taken, learner.action_space_size, align='mid')
        ax2.set_title("Histogram of actions taken (one game)")
        ax2.set_xlabel("Action")
        ax2.set_ylabel("Frequency")
        ax2.set_xticks((np.arange(0.4, (0.4 + learner.action_space_size - 0.75), step=0.75)))
        ax2.set_xticklabels(list(learner.game.get_action_meanings()))
        plt.show()
