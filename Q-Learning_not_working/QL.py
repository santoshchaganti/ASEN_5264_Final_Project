import gymnasium as gym
import numpy as np
from collections import defaultdict
import random


ALPHA = 0.1      
GAMMA = 0.99     
EPSILON = 1.0  
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
NUM_EPISODES = 1000
BUCKETS = (20, 20, 20)  


env = gym.make('CarRacing-v2', continuous=False)
ACTION_SPACE_SIZE = env.action_space.n


def discretize_state(state):
    
    high = env.observation_space.high
    low = env.observation_space.low
    ratios = [(state[i] - low[i]) / (high[i] - low[i]) for i in range(len(state))]
    buckets = [int(ratio * (BUCKETS[i] - 1)) for i, ratio in enumerate(ratios)]
    return tuple(min(max(b, 0), BUCKETS[i] - 1) for i, b in enumerate(buckets))


class QLearningAgent:
    def __init__(self, action_size):
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.epsilon = EPSILON

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SPACE_SIZE - 1)
        return np.argmax(self.q_table[state])


    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + GAMMA * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += ALPHA * (target - self.q_table[state][action])


agent = QLearningAgent(ACTION_SPACE_SIZE)

for episode in range(NUM_EPISODES):
    state = discretize_state(env.reset())
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
