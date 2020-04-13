# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:44:52 2020

@author: Abhishek
"""
import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

class GymAI:

    def __init__(self, game_env, episode, score_required):
        self.env = game_env
        self.episode = episode
        self.iteration = 200
        self.game_memory = []
        self.previous_observation =[]
        self.training_data = []
        self.score_required = score_required
        self.accepted_scores = []
        self.env.reset()
        
    def get_game_model(self):
        self.accepted_scores = []
        for game_index in range(self.episode):
            print(game_index)
            score = 0
            self.game_memory = []
            self.previous_observation = []
            for step_index in range(self.iteration):
                action = random.randrange(0, self.env.action_space.n)
                observation, reward, done, info = self.env.step(action)
                
                if len(self.previous_observation) > 0:
                    self.game_memory.append([self.previous_observation, action])
                    
                self.previous_observation = observation
                score += reward
                if done:
                    break
                
            if score >= self.score_required:
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
        X = np.array([i[0] for i in self.training_data]).reshape(-1, len(self.training_data[0][0]))
        y = np.array([i[1] for i in self.training_data]).reshape(-1, len(self.training_data[0][1]))
        model = self.build_model(input_size=len(X[0]), output_size=len(y[0]))    
        model.fit(X, y, epochs=5)
        print(model)
        return model
    
    def build_model(self,input_size, output_size):
        model = Sequential()
        model.add(Dense(150, input_dim=input_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

        