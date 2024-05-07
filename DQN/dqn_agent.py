"""
Module containing the implementation of a DQN Agent for the OpenAI Gym CarRacing environment.

This module defines the `DQNAgent` class, which is responsible for:
- Managing the replay buffer for experience replay.
- Training the agent using Deep Q-Learning (DQN).
- Predicting the optimal actions based on the agent's policy network.

Classes:
    DQNAgent: A class that manages training and evaluation of a DQN agent.

Functions:
    None
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random 
from collections import namedtuple, deque 
import gymnasium as gym
from model import DQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """
    A Deep Q-Network (DQN) Agent implementation.

    This class provides methods to initialize, train, and interact with the agent.

    Attributes:
        state_size (int): Size of the input state vector.
        action_size (int): Number of possible actions.
        memory (deque): Replay buffer storing past experiences.
        gamma (float): Discount factor for Q-learning.
        epsilon (float): Exploration rate for epsilon-greedy strategy.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for reducing exploration over time.
        learning_rate (float): Learning rate for the optimizer.
        model (keras.Model): The neural network model for Q-learning.
        target_model (keras.Model): The target network model for stable training.

    Methods:
        remember(state, action, reward, next_state, done): Store an experience in the replay buffer.
        act(state): Choose an action using an epsilon-greedy strategy.
        replay(batch_size): Train the model using experiences sampled from the replay buffer.
        update_target_model(): Update the target model weights to match the main model.
        load(name): Load a trained model from the specified file.
        save(name): Save the current model to the specified file.
    """
    def __init__(self):
        self.qnetwork_local = DQNetwork().to(device)
        self.qnetwork_target = DQNetwork().to(device)
        self.memory = deque(maxlen=10000) 
        self.gamma = 0.97 
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.batch_size = 64
        self.train_start = 3000
        
        self.count_1 = 0
        self.count_2 = 0

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=0.001)
        
    def step(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.count_2 = (self.count_2+1) % 500
        self.count_1 = (self.count_1+1) % 4

        if self.count_2 == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        if self.count_1 == 0 and len(self.memory) >= self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
            self.learn(minibatch)
            
    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        sample = random.random()
        if sample > self.epsilon:
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            action =  random.choice(np.arange(5))
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, batch):

        criterion = torch.nn.MSELoss()

        states =  np.zeros((self.batch_size, 96, 96 ,3))
        next_states =  np.zeros((self.batch_size, 96, 96 ,3))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            state_i, action_i, reward_i, next_state_i, done_i = batch[i]
            states[i] = state_i
            next_states[i] = next_state_i  
            actions.append(action_i)
            rewards.append(reward_i)
            dones.append(done_i)
        

        actions = np.vstack(actions).astype(np.int)
        actions = torch.from_numpy(actions).to(device)

        rewards = np.vstack(rewards).astype(np.float)
        rewards = torch.from_numpy(rewards).to(device)

        dones = np.vstack(dones).astype(np.int)
        dones = torch.from_numpy(dones).to(device)



        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        predictions = self.qnetwork_local(torch.from_numpy(states).float().to(device)).gather(1,actions)

        with torch.no_grad():
            q_next = self.qnetwork_target(torch.from_numpy(next_states).float().to(device)).detach().max(1)[0].unsqueeze(1)
        
        targets = rewards + (self.gamma * q_next * (1-dones))
        targets = targets.float()
        loss = criterion(predictions,targets).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()