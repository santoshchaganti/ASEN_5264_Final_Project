"""
Module to train and evaluate a DQN Agent in the OpenAI Gym CarRacing environment.

This module handles the main control flow for initializing and training a DQN agent.

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
from dqn_agent import DQNAgent
import matplotlib.pylab as plt


env = gym.make("CarRacing-v2",continuous=False)
env.observation_space
env.action_space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQNAgent()
rewards = [] 
total_episodes = 500
max_steps = 1000

"""
    Train a DQN agent in an OpenAI Gym environment, plot learning curves, and save models.

    This function executes the following steps:
    1. Train a DQN agent for a specified number of episodes while collecting rewards per episode.
    2. Plot and save the learning curve of agent scores over time.
    3. Save the agent's model weights after the training.
    4. Conduct further training for an additional 500 episodes to search for the best-performing model.
    5. Save the best model encountered during the search.

    Steps:
        - Initialize the environment and agent.
        - Loop through training episodes:
            - Reset the environment and track cumulative rewards.
            - Take actions based on the agent's policy and update state transitions.
            - Store the cumulative reward for each episode.
        - Save the rewards plot as a file.
        - Save the model weights.
        - Conduct further training to find the best-performing model.

    """

def main():

    for episode in range(total_episodes):
        state, _ = env.reset()
        cumulative_reward = 0 
        
        for i in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            cumulative_reward += reward
            if done:
                break
        
        rewards.append(cumulative_reward)
        print(f"Episode {episode}/{total_episodes}, Return = {cumulative_reward}, The epsilon now is : {agent.epsilon}")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rewards)),rewards)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    file_path = "/Users/santo/Documents/repo/DMU_Project/results/learning_curve.png"  # Change the file path as needed
    plt.savefig(file_path)
    print(f"Plot saved as {file_path}")

    torch.save(agent.qnetwork_target.state_dict(), '/Users/santo/Documents/repo/DMU_Project/results/weights.pth')
    
    best_reward = 700

    for episode in range(500):
        state, _ = env.reset()
        cumulative_reward = 0 
        
        for i in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            cumulative_reward += reward
            if done:
                break

            if cumulative_reward >= best_reward:
                best_reward = cumulative_reward
                torch.save(agent.qnetwork_target.state_dict(), '/Users/santo/Documents/repo/DMU_Project/results/weights_best.pt')
        
        rewards.append(cumulative_reward)
        print(f"Episode {episode}/{total_episodes}, Return = {cumulative_reward}, The epsilon now is : {agent.epsilon}")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rewards)),rewards)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()