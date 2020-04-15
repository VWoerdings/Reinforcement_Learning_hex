# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:44:52 2020

@author: Abhishek
"""
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

# https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2

class GymAI:

    def __init__(self, game_env, episode, required_score, steps, number_states):
        self.env = game_env
        self.episode = episode
        self.required_score = required_score
        self.steps = steps
        self.game_memory = []
        self.previous_observation = []
        self.accepted_scores = []
        self.training_data = []
        self.env.seed(0)
        np.random.seed(0)
        self.env.reset()
        
    def data_preparation(self):
        self.game_memory = []
        self.previous_observation = []
        self.accepted_scores = []
        self.training_data = []
        for game_index in range(self.episode):
            score = 0
            for index in range(self.steps):
                action = random.randrange(0, self.env.action_space.n)
                observation, reward, done, info = self.env.step(action)
                if len(self.previous_observation) > 0:
                    self.game_memory.append([self.previous_observation, action])
                self.previous_observation = observation
                if observation[0] > -0.2:
                    reward = 1
                score += reward
                if done:
                    break
            if score >= self.required_score:
                self.accepted_scores.append(score)
                for data in self.game_memory:
                    if data[1] == 1:
                        output = [0, 1, 0]
                    elif data[1] == 0:
                        output = [1, 0, 0]
                    elif data[1] == 2:
                        output = [0, 0, 1]
                    self.training_data.append([data[0], output])
            self.env.reset()
        print(self.accepted_scores)
        return self.training_data
    
    def build_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(128, input_dim=input_size, activation='relu'))
        model.add(Dense(52, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
    
    def train_model(self, data):
        X = np.array([i[0] for i in data]).reshape(-1, len(data[0][0]))
        y = np.array([i[1] for i in data]).reshape(-1, len(data[0][1]))
        model = self.build_model(len(X[0]),len(y[0]))
        model.fit(X, y, epochs=5)
        return model
    