import numpy as np
import matplotlib.pyplot as plt
import gym
import random

# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

class SARSALearningParams:
    def __init__(self, use_dynamic_lr=False):
        self.total_episodes = 15000
        self.learning_rate_init = 0.8
        self.learning_rate = 0.8
        self.learning_rate_min = 0.01
        self.learning_rate_decay = 0.001
        self.max_steps = 100
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.001
        self.use_dynamic_learning_rate = use_dynamic_lr

# Policy gradient action selection
def choose_action_policy_gradient(state, qtable):
    state_index = state
    action_probs = np.exp(qtable[state_index]) / np.sum(np.exp(qtable[state_index]))
    return np.random.choice(np.arange(action_size), p=action_probs)

# Random action selection for ALG-2
def choose_action_random(state, qtable):
    return np.random.randint(action_size)

# Epsilon-greedy action selection
def choose_action_epsilon_greedy(state, qtable, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(qtable[state, :])

# Common Q-table update function, adaptable for strategies
def update_qtable(state, action, reward, new_state, params, qtable):
    best_next_action = np.argmax(qtable[new_state])
    td_target = reward + params.discount_factor * qtable[new_state, best_next_action]
    td_error = td_target - qtable[state, action]
    qtable[state, action] += params.learning_rate * td_error

# Training agent function
def train_agent(params, choose_action_func, update_qtable_func, strategy_name):
    qtable = np.zeros((state_size, action_size))
    all_rewards = []
    epsilon = params.epsilon

    for episode in range(params.total_episodes):
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        total_rewards = 0

        for step in range(params.max_steps):
            if strategy_name == "epsilon-greedy":
                action = choose_action_epsilon_greedy(state, qtable, epsilon)
            else:
                action = choose_action_func(state, qtable)
            new_state, reward, terminated, truncated, _ = env.step(action)

            update_qtable_func(state, action, reward, new_state, params, qtable)

            state = new_state
            total_rewards += reward
            done = truncated or terminated

            if done:
                break

        # Epsilon decay for epsilon-greedy strategy
        if strategy_name == "epsilon-greedy" and params.use_dynamic_learning_rate:
            epsilon = max(params.min_epsilon, epsilon * np.exp(-params.decay_rate * episode))

        all_rewards.append(total_rewards)

    # Apply smoothing to rewards for plotting
    smoothed_rewards = smooth_rewards(all_rewards, window_size=100)
    return smoothed_rewards

# Smooth rewards for better visualization
def smooth_rewards(rewards, window_size=100):
    smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    return smoothed

def plot_learning_curves(smoothed_rewards_list, labels, title):
    plt.figure(figsize=(12, 8))
    for smoothed_rewards, label in zip(smoothed_rewards_list, labels):
        plt.plot(smoothed_rewards, label=label, linewidth=2)
    plt.xlabel('Episodes (Smoothed)', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# Parameters initialization
params = SARSALearningParams(use_dynamic_lr=True)

# Training with different strategies
smoothed_epsilon_greedy = train_agent(params, choose_action_epsilon_greedy, update_qtable, "epsilon-greedy")
smoothed_alg1 = train_agent(params, choose_action_policy_gradient, update_qtable, "ALG-1")
smoothed_alg2 = train_agent(params, choose_action_random, update_qtable, "ALG-2")

# Plotting learning curves
plot_learning_curves(
    [smoothed_epsilon_greedy, smoothed_alg1, smoothed_alg2],
    ['Epsilon-Greedy', 'ALG-1 (Policy Gradient)', 'ALG-2 (Random)'],
    'Comparison of Learning Strategies on FrozenLake'
)
