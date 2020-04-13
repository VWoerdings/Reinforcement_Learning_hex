# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:02:34 2020

@author: Abhishek
"""

from GymAI import *

env = gym.make("MountainCar")
episode = 1000
required_Score = -198

gymAI = GymAI(env, episode, required_Score)
model = gymAI.get_game_model()

env.reset()

scores = []
choices = []
for each_game in range(100):
    score = 0
    prev_obs = []
    for step_index in range(1000):
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0, env.action_space.n)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break
    env.reset()
    scores.append(score)

print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
