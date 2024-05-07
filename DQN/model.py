"""
Module defining neural network architectures for the DQN Agent.

This module provides a factory function to build a deep learning model.

Classes:
    DQNNetwork: This Class contains the neural network architecture for the DQN Agent.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random 
from collections import namedtuple, deque 
import gymnasium as gym

class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2) 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2) 
        self.fc1 = nn.Linear(128 * 4 * 4, 100) 
        self.fc2 = nn.Linear(100, 5)         

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2)) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(-1, 128 * 4 * 4) 
        x = F.relu(self.fc1(x))
        return self.fc2(x) 