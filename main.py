import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt


class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, initial_balance=1000):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.initial_balance = initial_balance
        self.current_step = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.stock_data.iloc[self.current_step]['Open'],
            self.stock_data.iloc[self.current_step]['High'],
            self.stock_data.iloc[self.current_step]['Low'],
            self.stock_data.iloc[self.current_step]['Close'],
            self.stock_data.iloc[self.current_step]['Volume'],
            self.balance
        ])
        return obs

    def step(self, action):
        current_price = self.stock_data.iloc[self.current_step]['Close']
        self.current_step += 1

        if action == 1:  # Buy
            self.shares_held += self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        done = self.current_step >= len(self.stock_data) - 1
        reward = self.balance + self.shares_held * current_price - self.initial_balance

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares Held: {self.shares_held}')
        print(f'Current Price: {self.stock_data.iloc[self.current_step]["Close"]}')

import yfinance as yf

# Download stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
stock_data.reset_index(inplace=True)

env = StockTradingEnv(stock_data)

# Q-Learning parameters
num_episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

# Initialize Q-table
num_states = 10  # Simplified state space
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))


# Discretize the state space
def discretize_state(state):
    return int(np.mean(state) * num_states) % num_states


# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    state = discretize_state(state)
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f'Episode {episode}, Epsilon: {epsilon}, Total Reward: {reward}')

state = env.reset()
state = discretize_state(state)
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])
    next_state, reward, done, _ = env.step(action)
    next_state = discretize_state(next_state)
    total_reward += reward
    state = next_state

print(f'Total Reward: {total_reward}')

plt.plot(stock_data['Close'])
plt.title('Stock Price')
plt.show()