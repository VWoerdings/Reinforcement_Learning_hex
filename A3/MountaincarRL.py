# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:02:34 2020

@author: Abhishek
"""
import gym
from GymAI import *

env = gym.make("MountainCar-v0")
episode = 1000
required_score = -198
steps = 200
number_states = 40

gymAI = GymAI(env, episode, required_score, steps, number_states)
train_data = gymAI.data_preparation()
model = gymAI.train_model(train_data)

scores = []
choices = []
for each_game in range(100):
    score = 0
    game_memory = []
    prev_obs = []
    for step_index in range(steps):
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
env.reset()
scores.append(score)
print(scores)
print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{} choice 2:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(2)/len(choices)))