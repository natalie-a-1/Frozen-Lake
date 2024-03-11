import gym
import random
import numpy as np
import matplotlib.pyplot as plt

class QLearningParams:
    def __init__(self, use_dynamic_lr=False):
        self.total_episodes = 25000
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

# Initialize the environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(qtable[state, :])
    return action

def update_qtable(state, action, reward, new_state, discount_factor, learning_rate):
    qtable[state, action] += learning_rate * (reward + discount_factor * np.max(qtable[new_state, :]) - qtable[state, action])

def train_qlearning(params):
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
            update_qtable(int(state), action, reward, int(new_state), params.discount_factor, params.learning_rate)
            
            total_rewards += reward
            state = new_state

            if done:
                break
        params.epsilon = params.min_epsilon + (params.max_epsilon - params.min_epsilon)*np.exp(-params.decay_rate*episode) 
        # params.epsilon = decay_epsilon(params.epsilon, episode, params.min_epsilon, params.max_epsilon, params.decay_rate, params.total_episodes)
        rewards.append(total_rewards)
    return rewards

def plot_moving_averages(rewards_list, window_size, labels, title):
    plt.figure(figsize=(12, 8))
    markers = ['o', 'o']  # Different markers for each plot
    colors = ['b', 'g']  # Color options
    for rewards, label, marker, color in zip(rewards_list, labels, markers, colors):
        moving_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg_rewards, label=label, marker=marker, linewidth=2, color=color, markevery=1000)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

window_size = 1000 
# Train Q-learning with constant learning rate
params_constant_lr = QLearningParams(use_dynamic_lr=False)
rewards_constant_lr = train_qlearning(params_constant_lr)

# Train Q-learning with dynamic learning rate
params_dynamic_lr = QLearningParams(use_dynamic_lr=True)
rewards_dynamic_lr = train_qlearning(params_dynamic_lr)

print(f"Average score over time: \nConstant Learning Rate:{np.mean(rewards_constant_lr)} \nDynamic Learning Rate:{np.mean(rewards_dynamic_lr)}")

# Assuming rewards_constant_lr and rewards_dynamic_lr are already computed
plot_moving_averages(
    [rewards_constant_lr, rewards_dynamic_lr], 
    window_size, 
    ['Constant Learning Rate', 'Dynamic Learning Rate'], 
    'Q-Learning Training Progress Comparison'
)