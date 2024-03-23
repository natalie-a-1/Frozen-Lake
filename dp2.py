#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import random
import gym
from gym import envs
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint


# In[33]:


# env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=True)
env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset()
env.render()


# In[34]:


state = env.reset()
num_states = env.observation_space.n
num_actions = env.action_space.n
print(num_states, ", ", num_actions)

new_state, reward, terminated, truncated, info = env.step(1)  
print(new_state)
print(reward)
print(info)


# In[39]:


pprint(env.env.P[5])


# In[40]:


threshold = 0.0001      # convergence threshold
gamma = 0.99            # discount rate


# In[41]:


def policy_evaluation(env, policy, gamma, threshold):
    num_states = env.observation_space.n
    V = np.zeros(num_states)
    max_delta = threshold + 1
    while max_delta > threshold:
        temp = np.zeros(num_states)
        for state in range(num_states):
            action = policy[state]
            for proba, new_state, reward, _ in env.env.P[state][action]:
                temp[state] += proba * (reward + V[new_state] * gamma)
        max_delta = np.max(np.abs(V - temp))
        V = np.copy(temp)
    return V


# In[42]:


def policy_improvement(env, V, gamma):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.zeros(num_states)
    for state in range(num_states):
        actions = np.zeros(num_actions)
        for action in range(num_actions):
            for proba, new_state, reward, _ in env.env.P[state][action]:
                actions[action] += proba * (reward + V[new_state] * gamma)
        policy[state] = np.argmax(actions)
    return policy


# In[43]:


def policy_iteration(env, gamma=0.99, threshold=0.0001):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.random.randint(low=0, high=num_actions, size=(num_states,)).astype(float)
    while True:
        V = policy_evaluation(env, policy, gamma=gamma, threshold=threshold)
        new_policy = policy_improvement(env, V, gamma=gamma)
        if np.array_equal(new_policy, policy):
            return V, new_policy
        policy = np.copy(new_policy)


# In[134]:


# def run_episode(environment, n_episodes, policy):
def run_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0
    for episode in range(n_episodes):
        
        terminated = False
        truncated = False
        
        state_info = environment.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        
        while not (terminated or truncated):
            action = np.argmax(policy[state])
#             print(f'episode {episode} not terminated or truncated')

            
            
            next_state, reward, terminated, truncated, info = environment.step(action)
            
            state = next_state
            
            if terminated or truncated:
#               This section of the code never executes...
#                 wins += 1
                total_reward += reward
                
    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward
                


# In[141]:


env = gym.make('FrozenLake-v1', is_slippery=True)
# env.reset()
# env.render()
V_optimal, optimal_policy = policy_iteration(env)
# total_reward = []
NUM_EPISODES = 1000
# for n in range(NUM_EPISODES):
#     total_reward.append(run_episode(env, optimal_policy))

wins, total, average = run_episodes(env, 1000, optimal_policy)
# print(f"Success rate over {NUM_EPISODES} episodes: {sum(total) * 100 / NUM_EPISODES}%")
print(f"Success rate over {NUM_EPISODES} episodes: {total * 100 / NUM_EPISODES}%")


# In[142]:


print(optimal_policy)
np.argmax(optimal_policy[3])


# In[117]:





# In[ ]:




