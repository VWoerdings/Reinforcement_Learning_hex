# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:44:52 2020

@author: Abhishek
"""

import gym
import tensorflow as tf
from collections import deque
import random
import numpy as np

class GymAI:
    """
        GymAI is the class to tain mountaincar using tensorflow. While inititating we pass
        game environment, episode numbers, number of steps and the buffer size.we initate
        all the variables and also initate a model value to one of the variable.
        there are two different models we can use prepare_model or prepare_model_2 
    """
    def __init__(self, env , episode, steps, buffer_size):
        self.env=env
        self.gamma=0.99
        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min=0.01
        self.learingRate=0.001
        self.game_memory=deque(maxlen=20000)
        self.trained_model = self.prepare_model()
        self.target_model = self.prepare_model()
        self.episode=episode
        self.steps=steps
        self.game_memory_buffer=buffer_size
        
        self.target_model.set_weights(self.trained_model.get_weights())
        self.graph_eps_vs_success = []
        self.graph_eps_vs_maxpos = []
        
    """
         Looping through the number of episodes and training the model
    """
    def initate(self):
        for eps in range(self.episode):
            current_state=self.env.reset().reshape(1,2)
            self.train_model(current_state, eps)
        return self.graph_eps_vs_success, self.graph_eps_vs_maxpos
    """
        We are creating MLP network with fully connected this is a model 1 where it has
        2 hidden layer with 50 and 25 neurons with an activation function on linear and
        optimizer ad Adam.
    """
    def prepare_model(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(tf.keras.layers.Dense(50, activation='relu', input_shape=state_shape))
        model.add(tf.keras.layers.Dense(25, activation='relu'))
        model.add(tf.keras.layers.Dense(self.env.action_space.n,activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learingRate))
        return model
    """
        this is a second model wehre we can compare how 
    """
    def prepare_model_2(self):
        model = tf.keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(tf.keras.layers.Dense(100, activation='relu', input_shape=state_shape))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(25, activation='relu'))
        model.add(tf.keras.layers.Dense(self.env.action_space.n,activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate))
        return model
    """
        This method return the best action that is avaliable in the model
    """
    def getBestAction(self,state):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action=np.argmax(self.trained_model.predict(state)[0])
        return action
    
    def train_model_with_buffer(self):
        if len(self.game_memory) < self.game_memory_buffer:
            return
        samples_list = random.sample(self.game_memory,self.game_memory_buffer)
        states_list = []
        new_states_list=[]
        for sample in samples_list:
            state, action, reward, new_state, done = sample
            states_list.append(state)
            new_states_list.append(new_state)
        newArray = np.array(states_list)
        states_list = newArray.reshape(self.game_memory_buffer, 2)
        newArray2 = np.array(new_states_list)
        new_states_list = newArray2.reshape(self.game_memory_buffer, 2)
        targets = self.trained_model.predict(states_list)
        new_state_targets=self.target_model.predict(new_states_list)
        i=0
        for sample in samples_list:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
            i+=1
        self.trained_model.fit(states_list, targets, epochs=1, verbose=0)


    def train_model(self,current_state,eps):
        rewardSum = 0
        max_position=-99
        for i in range(self.steps):
            bestAction = self.getBestAction(current_state)
            #show the animation every 50 eps
            if eps%50==0:
                self.env.render()
            new_state, reward, done, info = self.env.step(bestAction)
            new_state = new_state.reshape(1, 2)
            # # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]
            # # Adjust reward for task completion
            if new_state[0][0] >= 0.5:
                reward += 10
            self.game_memory.append([current_state, bestAction, reward, new_state, done])
            self.train_model_with_buffer()
            rewardSum += reward
            current_state = new_state
            if done:
                break
        if i >= 199:
            print("Failed to finish task in epsoide {}".format(eps))
        else:
            print("Car reached the hill top in {} epsoide and used {} iterations!".format(eps, i))
        self.target_model.set_weights(self.trained_model.get_weights())
        self.graph_eps_vs_maxpos.append(max_position)
        self.graph_eps_vs_success.append(rewardSum)
        self.epsilon -= self.epsilon_decay
