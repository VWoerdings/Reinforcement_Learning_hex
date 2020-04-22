import tensorflow as tf
from tensorflow.keras import layers, models
import gym
import random
import time
import numpy as np

# RL Assigment 3: DQN Learning; Part 2: Atari Breakout
# April 2020
# Abishek Ekaanth, Virgil Woerdings, Ruben Walen
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
    def __init__(self, frame_size, resize_factor, n_actions, loss_function, optimiser):
        self.original_frame_size = frame_size
        self.resize_factor = resize_factor
        self.reduced_frame_size = np.array([self.original_frame_size[0] * resize_factor,
                                            self.original_frame_size[1] * resize_factor,
                                            self.original_frame_size[2]])
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

    def predictQVectorFromFrame(self, frames):
        if len(frames.shape) == 3:
            frames = np.reshape(frames, (1, frames.shape[0], frames.shape[1], frames.shape[2])) # single-image batch
        resized_images = tf.image.resize(frames, self.reduced_frame_size[0:2]) # resize images first
        return self.model.predict(resized_images)
        #values = frame
        #for layer in self.model._layers:
        #    values = layer(values) # propagate through the layer
        #return values # return the values in the output layer after propagating through all layers

    def fit(self, input_frames, output_matrix, batch_size, epochs):                
        resized_images = tf.image.resize(input_frames, self.reduced_frame_size[0:2]) # resize images first
        self.model.fit(input_frames, output_matrix, batch_size=batch_size, epochs=epochs)
        return

class BreakoutExperienceBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def put(self, experience):
        # Put an experience item in the buffer (the experience item may contain anything)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            to_replace = random.randrange(self.max_size)
            self.buffer[to_replace] = experience # random replacement

    def sample(self, n_samples):
        # Sample n_samples from the experience replay buffer
        if n_samples > self.max_size:
            raise Exception("@BreakoutExperienceBuffer.sample: n_samples is larger than buffer size!")

        experience_choices = random.sample(self.buffer, n_samples) # randomly pick n_samples without replacement
        return experience_choices

    def clear(self):
        # Wipe the buffer
        self.buffer = []
        return

class BreakoutExperienceTrajectoryBuffer:
    def __init__(self, max_size_games):
        self.max_size = max_size_games
        self.trajectories = []
        self.total_samples = 0
        self.buffer_size_per_trajectory = []

    def putNewGame(self):
        # Put an empty new game in the trajectories buffer
        # Return the index of the new game in the trajectories buffer list
        if len(self.trajectories) < self.max_size:
            self.trajectories.append([]) # add empty game list
            self.buffer_size_per_trajectory.append(0)
            return (len(self.trajectories) - 1)
        else:
            to_replace = random.randrange(self.max_size)
            self.total_samples -= len(self.trajectories[to_replace])
            self.trajectories[to_replace] = [] # random replacement
            self.buffer_size_per_trajectory[to_replace] = 0
            return to_replace

    def addExperienceToGame(self, experience, game_index, backpropagation_discount=1.00):
        # Put an experience sample in a game trajectory (append at end).
        # If the reward for that experience is positive, backpropagate the reward along the trajectory.
        if game_index >= len(self.trajectories):
            raise IndexError("@BreakoutExperienceTrajectoryBuffer.addExperienceToGame: game index out of range")

        self.trajectories[game_index].append(experience)
        reward = experience[2] # index 2 associated with reward
        if reward > 0: # backpropagate positive rewards along the trajectory TODO: cache discounted exponentials?
            for i, exp in reversed(list(enumerate(self.trajectories[game_index]))[0:-1]):
                discounted_reward = (backpropagation_discount)**(i + 1) * reward
                self.trajectories[game_index][i] += discounted_reward # increase associated reward
        self.total_samples += 1
        self.buffer_size_per_trajectory[game_index] += 1
        return

    def sample(self, n_samples, equalise_over_games=False):
        # Sample n_samples from the buffer, if equalise_over_games is True: sample with equal probability over games instead of over total samples
        samples = []
        if not equalise_over_games:
            for _ in range(n_samples):
                sample_index = random.randrange(self.total_samples)
                game_index = None
                current_total = 0
                for i in range(len(self.trajectories)):
                    ingame_index = sample_index - current_total
                    current_total += self.buffer_size_per_trajectory[i]
                    if sample_index < current_total: # the index is in this game
                        game_index = i
                        break
                samples.append(self.trajectories[game_index][ingame_index])                
        else:
            for _ in range(n_samples):
                game_index = random.randrange(len(self.trajectories))
                ingame_index = random.randrange(self.buffer_size_per_trajectory[game_index])
                samples.append(self.trajectories[game_index][ingame_index])
        return samples

class BreakoutDQNLearner:
    def __init__(self, buffer_size, cycles_per_network_transfer, discount_factor):
        self.buffer = BreakoutExperienceBuffer(buffer_size)
        self.n_updates_count = 0 # how many times the network(s) was updated
        self.cycles_per_network_transfer = cycles_per_network_transfer # after how many update cylces we update the target network...
        self.discount_factor = discount_factor
        #... with the prediction network weights

        self.game = gym.make('Breakout-v0')
        self.current_frame = self.game.reset()
        self.current_frame, _, self.game_over, _ = self.game.step(random.choice(range(1, self.game.action_space.n)))
        self.game_over = False
        self.last_frame_time = 0
        self.action_space_size = self.game.action_space.n

        # target and prediction networks separated to reduce target instability
        sgd = tf.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # stochastic gradient descent
        rms = tf.optimizers.RMSprop(learning_rate=0.01, rho=0.9) # RMSprop
        self.target_network = BreakoutNetwork(self.current_frame.shape, 1, self.action_space_size, "mean_squared_error", rms)
        self.prediction_network = BreakoutNetwork(self.current_frame.shape, 1, self.action_space_size, "mean_squared_error", rms)

        self.buffer_indices = {'start_frame': 0, 'action': 1, 'reward': 2, 'game_over': 3, 'result_frame': 4}

        print(">BreakoutDQNLearner: Q check (initial frame)")
        print(self.target_network.predictQVectorFromFrame(self.current_frame))
        print("Actions:", self.game.get_action_meanings())

    def getMostPrudentAction(self, strategy='epsilon-greedy', **kwargs):
        # Use a strategy function to determine the most prudent action given the current frame (state) and environment
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
        action = self.getMostPrudentAction(strategy, **kwargs)
        frame, reward, game_over, _ = self.game.step(action)

        tup = (self.current_frame, action, reward, game_over, frame)
        if not do_not_store:
            self.buffer.put(tup) # put new action tuple in the experience replay buffer
        self.current_frame = frame
        self.game_over = game_over
        if self.game_over:
            self.resetAndRandomNonZeroMove()
        return tup

    def updateNetwork(self, use_replay_buffer=True, nsamples_replay_buffer=1, train_batch_size='auto', epochs=1, experiences=None):
        if train_batch_size == 'auto':
            train_batch_size = nsamples_replay_buffer
        if use_replay_buffer:
            experience_batch = self.buffer.sample(nsamples_replay_buffer) # throws exception: buffer content too small
        else:
            experience_batch = experiences
            
        target_matrix = np.zeros((nsamples_replay_buffer, self.action_space_size)) # matrix of target Q values
        input_frames = [None for _ in range(nsamples_replay_buffer)] # the frames to use as inputs
        for i, exp in enumerate(experience_batch):
            target_matrix[i, :] = self.prediction_network.predictQVectorFromFrame(exp[self.buffer_indices['start_frame']]) # no loss for non-represented action
            if exp[self.buffer_indices['game_over']] == False: # not a game over state - add the max Q of the next frame to the action taken
                max_Q = np.max(self.target_network.predictQVectorFromFrame(exp[self.buffer_indices['result_frame']])) # from frame resulting from action
                target_matrix[i, exp[self.buffer_indices['action']]] += self.discount_factor * max_Q
            target_matrix[i, exp[self.buffer_indices['action']]] += exp[self.buffer_indices['reward']] # update with reward associated with that action in that sample
            input_frames[i] = exp[self.buffer_indices['start_frame']]

        self.prediction_network.fit(np.array(input_frames), target_matrix, train_batch_size, epochs)
        self.n_updates_count += 1
        if self.n_updates_count % self.cycles_per_network_transfer == 0: # transfer prediction network to train network
            self.target_network.model.set_weights(self.prediction_network.model.get_weights())
        return

    def render(self, frame_rate_mills):
        # Wait for a maximum of frame_rate_mills milliseconds per render cycle, draw the game screen with most recent action
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
        self.game.reset() # reset the game completely
        self.current_frame, _, self.game_over, _ = self.game.step(random.choice(range(1, self.game.action_space.n))) # prevent getting stuck in zero-moves
        return
        

if __name__ == "__main__":
    # DQN Learning on Atari Breakout
    BUFFER_SIZE = 1000
    CYCLES_FOR_TRANSFER = 10
    N_ACTIONS_PER_PLAY_CYCLE = 10
    N_SAMPLES_PER_LEARN_CYCLE = 25
    N_EPOCHS_PER_LEARN_CYCLE = 5
    N_CYCLES_PERFORMANCE_EVAL = 0
    N_EPOCHS_MASTER = 2500
    EPSILON = 0.7
    DISCOUNT = 1.00
    FRAME_RATE = 0.02
    #np.random.seed(333)
    #random.seed(333)
    
    learner = BreakoutDQNLearner(BUFFER_SIZE, CYCLES_FOR_TRANSFER, DISCOUNT)
    print(">__main__: Filling buffer (samples:", BUFFER_SIZE, "total)")
    for i in range(BUFFER_SIZE): # buffer filling
        #print("Filling buffer: cycle", i + 1)
        learner.takeActionAndStoreExperience(epsilon=EPSILON, strategy='random')
    #learner.render(FRAME_RATE)
    for i in range(N_EPOCHS_MASTER):
        print("Master epoch", i + 1)
        for _ in range(N_ACTIONS_PER_PLAY_CYCLE):
            learner.takeActionAndStoreExperience(epsilon=EPSILON)
            #learner.render(FRAME_RATE)
        learner.updateNetwork(nsamples_replay_buffer=N_SAMPLES_PER_LEARN_CYCLE, epochs=N_EPOCHS_PER_LEARN_CYCLE)
        total_score = 0
        state = learner.game.clone_full_state()
        for _ in range(N_CYCLES_PERFORMANCE_EVAL):
            learner.resetAndRandomNonZeroMove()
            tup = learner.takeActionAndStoreExperience(epsilon=EPSILON, do_not_store=True)
            total_score += tup[learner.buffer_indices['reward']]
        learner.game.restore_full_state(state)
        print("Total score for master epoch:", total_score)

    # Test the AI in NUM_GAMES games
    NUM_GAMES = 4
    learner.resetAndRandomNonZeroMove()
    total_score = 0
    game_score = 0
    games_completed = 0
    complete = False
    while not complete:
        tup = learner.takeActionAndStoreExperience(epsilon=0.9, do_not_store=True)
        #print("Action", tup[learner.buffer_indices['action']])
        learner.render(FRAME_RATE)
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
