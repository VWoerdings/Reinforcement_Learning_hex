# -*- coding: utf-8 -*-
"""
# RL Assigment 3: DQN Learning; Part 1: Mountain car
# April 2020
# Abhishek Sira chandrashekar, Virgil Woerdings, Ruben Walen
"""
import gym
from GymAI import *
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
episode = 400
steps = 200
buffer_size = 32
gymAI=GymAI(env , episode, steps, buffer_size)
graph_eps_vs_success, graph_eps_vs_maxpos = gymAI.initate()


plt.plot(graph_eps_vs_maxpos)
plt.xlabel('Episode')
plt.ylabel('Furthest Position')
plt.show()
    
plt.plot(graph_eps_vs_success)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


episode = 400
steps = 200
buffer_size = 55
gymAI=GymAI(env , episode, steps, buffer_size)
graph_eps_vs_success, graph_eps_vs_maxpos = gymAI.initate()


plt.plot(graph_eps_vs_maxpos)
plt.xlabel('Episode')
plt.ylabel('Furthest Position')
plt.show()
    
plt.plot(graph_eps_vs_success)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()