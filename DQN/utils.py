"""
Module containing utility functions for training and evaluating DQN agents.

This module defines helper functions for plotting, data transformation, and other utilities.

"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random 
from collections import namedtuple, deque 
import gymnasium as gym
from matplotlib import animation
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

def save_env_as_gif(frames, path = '/Users/santos/Documents/repo/DMU_Project/results', filename='car_racing_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

env2 = gym.make("CarRacing-v2",continuous=False, render_mode="rgb_array")
observation, _ = env2.reset()
frames = []
for t in range(1000):
    frames.append(env2.render())
    action = DQNAgent.act(observation)
    observation, _, done, _, _ = env2.step(action)
    if done:
        break
env2.close()
save_env_as_gif(frames)