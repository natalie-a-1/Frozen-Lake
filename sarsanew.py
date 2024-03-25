#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import gym


# In[26]:


class SARSALearningParams:
    def __init__(self, use_dynamic_lr=False):
        self.total_episodes = 15000
        self.learning_rate_init = 0.8  # Only for dynamic LR
        self.learning_rate = 0.8  # Constant learning rate
        self.learning_rate_min = 0.01  # Only for dynamic LR
        self.learning_rate_decay = 0.001  # Only for dynamic LR
        self.max_steps = 100
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.001
        self.use_dynamic_learning_rate = use_dynamic_lr
        
def choose_action_policy_gradient(state, qtable):
    state_index = state[0] if isinstance(state, tuple) else state  # Convert tuple to integer if necessary
    action_probs = qtable[state_index] / np.sum(qtable[state_index])  # Normalize Q-values to probabilities
    if np.isnan(action_probs).any():
        # If there are NaN values, reset probabilities to a uniform distribution
        action_probs = np.ones_like(action_probs) / len(action_probs)
    return np.random.choice(np.arange(len(qtable[state_index])), p=action_probs)

def update_qtable_policy_gradient(state, action, reward, new_state, discount_factor, learning_rate, qtable):
    qtable[state, action] += learning_rate * (reward + discount_factor * (qtable[new_state, choose_action_policy_gradient(new_state, qtable)]) - qtable[state, action])

def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(qtable[state, :])
    return action

def update_qtable(state, action, reward, new_state, discount_factor, learning_rate, epsilon):
    qtable[state, action] += learning_rate * (reward + discount_factor * (qtable[new_state, choose_action(new_state, epsilon)]) - qtable[state, action])

def plot_learning_curves(mean_rewards, labels, title):
    plt.figure(figsize=(12, 8))
    for mean, label in zip(mean_rewards, labels):
        plt.plot(mean, label=label, linewidth=2)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
def smooth_rewards(rewards, window_size=100):
    smoothed = []
    for i in range(len(rewards) - window_size + 1):
        window_avg = sum(rewards[i:i+window_size]) / window_size
        smoothed.append(window_avg)
    return smoothed


# In[33]:


def train_sarsa(params, runs=30):
    all_rewards = []
    for run in range(runs):  # Loop over multiple runs
        # Training loop
        global qtable
        qtable = np.zeros((state_size, action_size))
        rewards = []

        for episode in range(params.total_episodes):
            if params.use_dynamic_learning_rate:
                params.learning_rate = max(params.learning_rate_min, params.learning_rate_init - (episode / params.total_episodes) * (params.learning_rate_init - params.learning_rate_min))
            else:
                params.learning_rate = params.learning_rate

            state_info = env.reset()
            state = state_info[0] if isinstance(state_info, tuple) else state_info
            done = False
            total_rewards = 0

            for step in range(params.max_steps):
                action = choose_action_policy_gradient(state, qtable)  # Use policy gradient for action selection
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                update_qtable_policy_gradient(int(state), action, reward, int(new_state), params.discount_factor, params.learning_rate, qtable)  # Update Q-table using policy gradient
                
                total_rewards += reward
                state = new_state
                
                done = terminated or truncated

                if done:
                    break
            rewards.append(total_rewards)

        all_rewards.append(rewards)

    rewards_mean = np.mean(all_rewards, axis=0)
    rewards_std = np.std(all_rewards, axis=0)
    return rewards_mean, rewards_std


# In[34]:


def train_sarsa2(params, runs=30):

    all_rewards = []
    for run in range(runs):  # Loop over multiple runs
        # Training loop
        rewards = []

        for episode in range(params.total_episodes):
            if params.use_dynamic_learning_rate:
                params.learning_rate = max(params.learning_rate_min, params.learning_rate_init - (episode / params.total_episodes) * (params.learning_rate_init - params.learning_rate_min))
            else:
                params.learning_rate = params.learning_rate

            state_info = env.reset()
            state = state_info[0] if isinstance(state_info, tuple) else state_info
            done = False
            total_rewards = 0
            episode_states, episode_actions, episode_rewards = [], [], []

            for step in range(params.max_steps):
                action_probs = np.exp(qtable[state, :]) / np.sum(np.exp(qtable[state, :]))
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                episode_states.append(state)
                episode_actions.append(action)

                result = env.step(action)
                new_state, reward, terminated, truncated, _ = result if len(result) == 5 else result[0], result[1], result[2], result[3], {}
                new_state = new_state[0] if isinstance(new_state, tuple) else new_state
                episode_rewards.append(reward)

                if terminated or truncated:
                    break
                state = new_state

            # Calculate returns
            returns = np.cumsum(episode_rewards[::-1])[::-1]

            # Update policy parameters
            for t in range(len(episode_states)):
                state_t, action_t = episode_states[t], episode_actions[t]
                qtable[state_t, action_t] += params.learning_rate * (returns[t] - np.sum(returns[t:]) * np.exp(qtable[state_t, action_t]))

            total_rewards = sum(episode_rewards)
            rewards.append(total_rewards)

        all_rewards.append(rewards)

    rewards_mean = np.mean(all_rewards, axis=0)
    if runs > 1:
        # Calculate std across runs (makes sense only if runs > 1)
        rewards_std = np.std(np.mean(all_rewards, axis=1))  # Std dev of average rewards per run
    else:
        # Calculate std across episodes in the single run
        rewards_std = np.std(all_rewards) 
        
    return rewards_mean, rewards_std


# In[35]:


def train_sarsalearning(params, runs=30):

    all_rewards = []
    for run in range(runs):  # Loop over multiple runs
        # Training loop
        global qtable
        qtable = np.zeros((state_size, action_size))
        rewards = []

        for episode in range(params.total_episodes):
            if params.use_dynamic_learning_rate:
                params.learning_rate = max(params.learning_rate_min, params.learning_rate_init - (episode / params.total_episodes) * (params.learning_rate_init - params.learning_rate_min))
            else:
                params.learning_rate = params.learning_rate

            state_info = env.reset()
            state = state_info[0] if isinstance(state_info, tuple) else state_info
            done = False
            total_rewards = 0

            for step in range(params.max_steps):
                action = choose_action(state, params.epsilon)
                result = env.step(action)
                new_state, reward, terminated, truncated, _ = result if len(result) == 5 else result[0], result[1], result[2], result[3], {}
                new_state = new_state[0] if isinstance(new_state, tuple) else new_state
                
                done = terminated or truncated
                update_qtable(int(state), action, reward, int(new_state), params.discount_factor, params.learning_rate, params.epsilon)
                
                total_rewards += reward
                state = new_state

                if done:
                    break
            params.epsilon = params.min_epsilon + (params.max_epsilon - params.min_epsilon)*np.exp(-params.decay_rate*episode) 
        
            rewards.append(total_rewards)

        all_rewards.append(rewards)

    rewards_mean = np.mean(all_rewards, axis=0)
    if runs > 1:
        # Calculate std across runs (makes sense only if runs > 1)
        rewards_std = np.std(np.mean(all_rewards, axis=1))  # Std dev of average rewards per run
    else:
        # Calculate std across episodes in the single run
        rewards_std = np.std(all_rewards) 
    return rewards_mean, rewards_std


# In[36]:


env = gym.make("FrozenLake-v1")  # Assuming FrozenLake environment
state_size = env.observation_space.n
action_size = env.action_space.n


# In[37]:


params1 = SARSALearningParams(use_dynamic_lr=False)  # Set learning parameters
params2 = SARSALearningParams(use_dynamic_lr=False)  # Set learning parameters
params_constant_lr = SARSALearningParams(use_dynamic_lr=False)


# In[38]:


mean_rewards1, rewards_std1 = train_sarsa(params1)  # Train SARSA

env = gym.make('FrozenLake-v1', render_mode="rgb_array", desc=None, map_name="4x4", is_slippery=False)
env.reset()
env.render()
mean_rewards2, rewards_std2 = train_sarsa2(params2)  # Train SARSA

env.reset()
env.render()
rewards_mean_eg, std = train_sarsalearning(params_constant_lr, runs=30)

smoothed_constant1 = smooth_rewards(mean_rewards1, window_size=100) 
smoothed_constant2 = smooth_rewards(mean_rewards2, window_size=100) 

rewards_mean_eg, std = train_sarsalearning(params_constant_lr, runs=30)
smoothed_constant_eg = smooth_rewards(rewards_mean_eg, window_size=100)



# In[40]:


plot_learning_curves(
    [smoothed_constant1, smoothed_constant_eg],
    ['ALG-1', 'epsilon-greedy'],
    'SARSA Training Average Rewards'
)


# In[ ]:





# In[ ]:




