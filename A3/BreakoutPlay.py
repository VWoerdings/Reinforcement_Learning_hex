import tensorflow as tf
from tensorflow.keras import layers, models
import gym
import random
import time
import numpy as np

class BreakoutNetwork:
    def __init__(self, frame_size, resize_factor, n_actions, loss_function, optimiser):
        self.original_frame_size = original_frame_size
        self.resize_factor = resize_factor
        self.reduced_frame_size = np.array([self.original_frame_size[0] * resize_factor,
                                            self.original_frame_size[1] * resize_factor,
                                            self.original_frame_size[2]])
        self.n_actions = n_actions

        # construct the network
        self.network_params = [(32, (3, 3))] # convolutional feature dimensionality (output) and stride: every entry = 1 conv. layer

        model = models.Sequential()
        for i, entry in enumerate(self.network_params):
            model.add(layers.Conv2D(entry[0], entry[1], activation='linear', input_shape=self.reduced_frame_size)) # convolutional layer
            model.add(layers.MaxPooling2D((2, 2))) # max pooling to lower image size
        
        model.add(layers.Flatten()) # flatten feature maps
        model.add(layers.Dense(self.network_params[-1][0], activation='relu')) # first dense layer: connect to last conv. layer
        model.add(layers.Dense(n_actions)) # second dense layer: output actions

        self.model = model
        self.model.compile(loss=loss_function, optimizer=optimiser)

    def predictQVectorFromFrame(self, frames):
        resized_images = tf.image.resize(frames, self.reduced_frame_size[0:2]) # resize images first
        return self.model.predict(resized_images)
        #values = frame
        #for layer in self.model._layers:
        #    values = layer(values) # propagate through the layer
        #return values # return the values in the output layer after propagating through all layers

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

class BreakoutDQNLearner:
    def __init__(self, buffer_size, cycles_per_network_transfer):
        self.buffer = BreakoutExperienceBuffer(buffer_size)
        self.n_updates_count = 0 # how many times the network(s) was updated
        self.cycles_per_network_transfer = cycles_per_network_transfer # after how many update cylces we update the target network...
        #... with the prediction network weights

        self.game = gym.make('Breakout-v0')
        self.current_frame = self.game.reset()
        self.game_over = False
        self.last_frame_time = 0
        self.action_space_size = self.game.action_space.n

        # target and prediction networks separated to reduce target instability
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # stochastic gradient descent
        self.target_network = BreakoutNetwork(frame.shape, 1, self.action_space_size, "mean_squared_error", sgd)
        self.prediction_network = BreakoutNetwork(frame.shape, 1, self.action_space_size, "mean_squared_error", sgd)

        self.buffer_indices = {'start_frame': 0, 'action': 1, 'reward': 2, 'game_over': 3, 'result_frame': 4}

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
        action = self.getMostPrudentAction(strategy, kwargs)
        frame, reward, game_over, _ = self.game.step(action)

        tup = (self.current_frame, action, reward, game_over, frame)
        if not do_not_store:
            self.buffer.put(tup) # put new action tuple in the experience replay buffer
        self.current_frame = frame
        self.game_over = game_over
        if self.game_over:
            self.game.reset() # reset the game completely
        return tup

    def updateNetwork(use_replay_buffer=True, nsamples_replay_buffer=1, train_batch_size='auto', epochs=1, experiences=None):
        if train_batch_size = 'auto':
            train_batch_size = nsamples_replay_buffer
        if use_replay_buffer:
            experience_batch = self.buffer.sample(nsamples_replay_buffer) # throws exception: buffer content too small
        else:
            experience_batch = experiences
            
        target_matrix = np.zeros((nsamples_replay_buffer, self.action_space_size)) # matrix of target Q values
        input_frames = [None for _ in range(nsamples_replay_buffer)] # the frames to use as inputs
        for i, exp in enumerate(experience_batch):
            target_matrix[i, :] = self.target_network.predictQVectorFromFrame(exp[self.buffer_indices['result_frame']]) # from frame resulting from action
            target_matrix[i, exp[self.buffer_indices['action']]] += exp[self.buffer_indices['reward']] # update with reward associated with that buffer
            input_frames[i] = exp[self.buffer_indices['start_frame']]

        self.prediction_network.fit(input_frames, target_matrix, batch_size=train_batch_size, epochs=epochs)
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
            return random.randrange(Q_vector.shape[0])
        pass

    def _selectRandom(self, Q_vector):
        # Select a random action
        # Q_vector contains the predicted Q-value (value function) for each action
        # Returns an integer corresponding to the chosen action
        return random.randrange(Q_vector.shape[0])
        

if __name__ == "__main__":
    Breakout_game = gym.make('Breakout-v0')
    frame = Breakout_game.reset()
    Breakout_game.render()
    frame_time = 0.05 # minimum time per frame in seconds

    game_over = False
    while not game_over:
        time_start = time.time()
        action = Breakout_game.action_space.sample()
        print("Action:", Breakout_game.get_action_meanings()[action])
        frame, reward, game_over, _ = Breakout_game.step(action)
        print("Reward:", reward)
        Breakout_game.render()

        time.sleep(max(0, (frame_time - (time.time() - time_start))))

    print(frame.shape)
    bn = BreakoutNetwork(frame.shape, Breakout_game.action_space.n)
    print(bn.model.__dict__)
    for layer in bn.model._layers:
        print(layer.variables)
