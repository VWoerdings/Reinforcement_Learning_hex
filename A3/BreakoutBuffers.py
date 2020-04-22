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
        n_positive_samples = int(ratio_allocated_to_positive * n_samples)
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
        return experience_choices_main.extend(experience_choices_positive) # not shuffled!

    def clear(self):
        # Wipe the buffer(s)
        self.main_buffer = []
        self.positive_buffer = []
        return

class BreakoutExperienceTrajectoryBuffer:
    # A buffer that stores and samples from game trajectories instead of individual experiences
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

    def clear(self):
        self.trajectories = []
        self.total_samples = 0
        self.buffer_size_per_trajectory = []
        return
