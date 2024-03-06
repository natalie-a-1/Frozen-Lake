import gym
import random
import numpy as np
import matplotlib.pyplot as plt

class QLearningParams:
    total_episodes = 25000  # Adjusted for testing, increase for better learning
    learning_rate = 0.8
    max_steps = 100
    discount_factor = 0.95
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.001
    n_runs = 1  # Simplified to a single run for now

params = QLearningParams()

# Initialize the environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

qtable = np.zeros((state_size, action_size))

def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(qtable[state, :])
    return action

def update_qtable(state, action, reward, new_state, discount_factor, learning_rate):
    qtable[state, action] += learning_rate * (reward + discount_factor * np.max(qtable[new_state, :]) - qtable[state, action])

# Training loop
rewards = []

for episode in range(params.total_episodes):
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

print(f"Average score over time: {np.mean(rewards)}")
#print(qtable)

# Optionally, plot the moving average of rewards to see trends more clearly
window_size = 50  # Define the window size for the moving average
moving_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(12, 8))
    
# Plot the learning curve
ax.plot(moving_avg_rewards, label='Learning Performance', linewidth=2)
ax.set_xlabel('Episodes', fontsize=14)
ax.set_ylabel('Reward', fontsize=14)
ax.set_title('Q-Learning Training Progress', fontsize=16)
ax.legend(loc='upper left')  # Place legend on the upper left of the graph
ax.grid(True)  # Add grid for better readability

plt.tight_layout()  # Adjust the padding between and around subplots
plt.show()