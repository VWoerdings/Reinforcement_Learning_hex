import random
import numpy as np

# RL Assigment 3: DQN Learning; Part 2: Atari Breakout
# April 2020
# Abishek Ekaanth, Virgil Woerdings, Ruben Walen
# BreakoutBuffers.py: contains experience buffers
#
# ENHANCEMENTS:
# trajectories buffer: instead of storing individual samples...
#... store whole-game trajectories. Backpropagate (discounted) rewards...
#... along the game trajectory when a positive reward is received
# main/positive buffer split: always attempt to train network with a certain...
#... proportion of positive reward samples kept separately in the buffer

class BreakoutExperienceBuffer:
    # A basic experience buffer: has a maximum size, supports random putting and sampling
    def __init__(self, max_size):
        # Params:
        # max_size (int): the maximum size of the buffer
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
        if n_samples > len(self.buffer):
            raise Exception("@BreakoutExperienceBuffer.sample: n_samples is larger than buffer size!")
        if len(self.buffer) <= 0:
            raise Exception("@BreakoutExperienceBuffer.sample: the buffer is empty!")

        experience_choices = random.sample(self.buffer, n_samples) # randomly pick n_samples without replacement
        return experience_choices

    def clear(self):
        # Wipe the buffer
        self.buffer = []
        return

class BreakoutExperiencePosisplitBuffer:
    # An experience buffer that contains an additional buffer for positive reward samples.
    # During sampling, part of the n_samples can be allocated to positive reward samples only, if possible.
    def __init__(self, max_size_main, max_size_positive):
        # Params:
        # max_size_main (int): the maximum size of the main buffer
        # max_size_positive (int): the maximum size of the positive samples buffer
        self.max_size_main = max_size_main
        self.max_size_positive = max_size_positive
        self.main_buffer = []
        self.positive_buffer = []

    def put(self, experience):
        # Put an experience item in the main buffer (the experience item may contain anything)
        # Also adds to the positive buffer if reward > 0
        if len(self.main_buffer) < self.max_size_main:
            self.main_buffer.append(experience)
        else:
            to_replace = random.randrange(self.max_size_main)
            self.main_buffer[to_replace] = experience # random replacement

        reward = experience[2] # index 2 associated with reward
        if reward > 0: # add the sample to the positive buffer if reward > 0
            if len(self.positive_buffer) < self.max_size_positive:
                self.positive_buffer.append(experience)
            else:
                to_replace = random.randrange(self.max_size_positive)
                self.positive_buffer[to_replace] = experience # random replacement

    def sample(self, n_samples, part_allocated_to_positive=0.2):
        # Sample n_samples from the buffer. A cut of the n_samples can be drawn from the positive buffer instead.
        # If there are not enough samples in the positive buffer, the rest is drawn from the main buffer.
        if len(self.main_buffer) <= 0:
            raise Exception("@BreakoutExperiencePosisplitBuffer.sample: the main buffer is empty!")
        
        n_positive_samples = int(part_allocated_to_positive * n_samples)
        n_main_samples = n_samples - n_positive_samples
        positive_sampling_deficit = len(self.positive_buffer) - n_positive_samples
        if positive_sampling_deficit < 0:
            n_positive_samples = len(self.positive_buffer)
            n_main_samples = n_main_samples - positive_sampling_deficit # -- = +
        if n_main_samples > len(self.main_buffer):
            raise Exception("@BreakoutExperiencePosisplitBuffer.sample: n_samples is larger than combined possible main/positive buffer size!")

        experience_choices_main = []
        if n_main_samples > 0:
            experience_choices_main = random.sample(self.main_buffer, n_main_samples) # randomly pick samples without replacement
        experience_choices_positive = []
        if n_positive_samples > 0:
            experience_choices_positive = random.sample(self.positive_buffer, n_positive_samples)
        experience_choices_main.extend(experience_choices_positive) # not shuffled!
        return experience_choices_main

    def clear(self):
        # Wipe the buffer(s)
        self.main_buffer = []
        self.positive_buffer = []
        return

class BreakoutExperienceTrajectoryBuffer:
    # A buffer that stores and samples from game trajectories instead of individual experiences
    def __init__(self, max_size_games, auto_backpropagation_discount=1.00):
        # Params:
        # max_size_games (int): the maximum size in games of the buffer
        # auto_backpropagation_discount (float, 0 to 1): the discount factor for the automatic putter put()
        self.max_size = max_size_games
        self.trajectories = []
        self.total_samples = 0
        self.buffer_size_per_trajectory = []
        self.active_index = None # used for automatic putting (see function self.put)
        self.auto_backpropagation_discount = auto_backpropagation_discount # used in the automatic putter

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
        # Put an experience sample in a game trajectory at game_index (append at end).
        # backpropagation_discount is the backpropagated discount along the rest of the samples
        # If the reward for that experience is positive, backpropagate the reward along the trajectory.
        if game_index >= len(self.trajectories):
            raise IndexError("@BreakoutExperienceTrajectoryBuffer.addExperienceToGame: game index out of range")

        self.trajectories[game_index].append(experience)
        reward = experience[2] # index 2 associated with reward
        if reward > 0: # backpropagate positive rewards along the trajectory TODO: cache discounted exponentials?
            for i, exp in reversed(list(enumerate(self.trajectories[game_index]))[0:-1]):
                discounted_reward = (backpropagation_discount)**(i + 1) * reward
                unpack_list = list(self.trajectories[game_index][i]) # stupid tuples
                unpack_list[2] += discounted_reward # increase reward; index 2 associated with reward
                self.trajectories[game_index][i] = tuple(unpack_list)
        self.total_samples += 1
        self.buffer_size_per_trajectory[game_index] += 1
        return

    def sample(self, n_samples, equalise_over_games=False):
        # Sample n_samples from the buffer, if equalise_over_games is True: sample with equal probability over games instead of over total samples
        # equalise_over_games (bool): whether to uniformly sample a game to pick from before drafting a sample
        if n_samples > self.total_samples:
            raise Exception("@BreakoutExperienceTrajectoryBuffer.sample: n_samples is larger than buffer size!")
        if self.total_samples <= 0:
            raise Exception("@BreakoutExperienceTrajectoryBuffer.sample: the buffer is empty!")
        
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
                while True:
                    game_index = random.randrange(len(self.trajectories))
                    if self.buffer_size_per_trajectory[game_index] > 0: # prevent zero-drafting
                        break
                ingame_index = random.randrange(self.buffer_size_per_trajectory[game_index])
                samples.append(self.trajectories[game_index][ingame_index])
        return samples

    def put(self, experience, force_new_game=False):
        # This is an automatic buffer putter.
        # It adds new games and puts experiences in the 'current' game automatically.
        # force_new_game (bool): whether to force a new game to be created in the buffer in the current put() cycle
        if (len(self.trajectories) == 0) or force_new_game:
            self.active_index = None
        if self.active_index == None:
            self.active_index = self.putNewGame() # create a new game
            
        game_over = experience[3] # index 3 associated with game over
        self.addExperienceToGame(experience, self.active_index, backpropagation_discount=self.auto_backpropagation_discount)
        if game_over:
            self.active_index = None # create a new game next put cycle, the last game was game over
        return

    def clear(self):
        # Wipe the buffer
        self.trajectories = []
        self.total_samples = 0
        self.buffer_size_per_trajectory = []
        self.active_index = None
        return
