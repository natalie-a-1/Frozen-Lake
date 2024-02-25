import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(fixed_lr_rewards, dynamic_lr_rewards, num_episodes):
    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the learning curves
    ax.plot(fixed_lr_rewards, label='Fixed Learning Rate', linewidth=2)
    ax.plot(dynamic_lr_rewards, label='Dynamic Learning Rate', linewidth=2)
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=14)
    ax.set_title('Q-Learning Performance: Fixed vs Dynamic Learning Rate', fontsize=16)
    ax.legend(loc='upper left')  # Place legend on the upper left of the graph
    ax.grid(True)  # Add grid for better readability

    plt.tight_layout()  # Adjust the padding between and around subplots
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom to make room for the table
    plt.show()

def epsilon_greedy_policy(Q, state, epsilon):
    """
    Chooses an action based on the epsilon-greedy policy.

    Parameters:
    - Q (nparray): The Q-table, storing Q-values for state-action pairs.
    - state (int): The current state of the agent in the environment.
    - epsilon (float): The probability of choosing a random action, facilitating exploration.

    Returns:
    - int: The chosen action.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])

def dynamic_learning_rate(initial_rate, episode, decay_rate):
    """
    Calculates a dynamic learning rate based on the episode number.

    Parameters:
    - initial_rate (float): The initial learning rate.
    - episode (int): The current episode number.
    - decay_rate (float): The rate at which the learning rate decays over episodes.

    Returns:
    - float: The adjusted learning rate.
    """
    return initial_rate / (1 + decay_rate * episode)

def q_learning(env, num_episodes, learning_rate=0.8, discount_factor=0.99, epsilon=1.0, dynamic_lr=False):
    """
    Implements Q-learning with an option for dynamic learning rate.

    Parameters:
    - env: The environment.
    - num_episodes (int): Number of episodes for training.
    - learning_rate (float): The initial learning rate, used as fixed if dynamic_lr is False.
    - discount_factor (float): The discount factor for future rewards.
    - epsilon (float): The exploration rate.
    - dynamic_lr (bool): Whether to use dynamic learning rate.

    Returns:
    - ndarray: The learned Q-table.
    """
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    lr = learning_rate
    for episode in range(num_episodes):
        if dynamic_lr:
            lr = dynamic_learning_rate(learning_rate, episode, decay_rate=0.005)
        state, _ = env.reset()
        done = False
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state, :])
            td_target = reward + discount_factor * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += lr * td_error
            state = next_state
    return Q


def evaluate_policy(env, Q, num_episodes):
    """
    Evaluates the learned policy by running it without exploration.

    Parameters:
    - env: The environment conforming to the OpenAI Gym interface.
    - Q (ndarray): The learned Q-table.
    - num_episodes (int): Number of episodes for evaluation.

    Returns:
    - float: The average reward per episode.
    """
    total_reward = 0
    policy = np.argmax(Q, axis=1)
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy[observation]
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes

def demo_agent(env, Q, learning_rate_type, num_episodes=1):
    """
    Demonstrates the agent's performance with the learned policy.
    
    Parameters:
    - env: The environment instance for demonstration.
    - Q (ndarray): The learned Q-table.
    - learning_rate_type (str): Specifies the type of learning rate ('fixed' or 'dynamic').
    - num_episodes (int): Number of episodes to demonstrate.
    """
    policy = np.argmax(Q, axis=1)
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        print(f"\nEpisode: {episode + 1} | Learning Rate Type: {learning_rate_type.capitalize()}")
        while not done:
            env.render()
            action = policy[observation]
            observation, _, done, _, _ = env.step(action)
        env.render()

def collect_rewards(env, Q, num_episodes, interval=100):
    """
    Collects average rewards over specified intervals for plotting.

    Parameters:
    - env: The environment.
    - Q: The Q-table.
    - num_episodes: Total number of episodes for which to collect rewards.
    - interval: The interval over which to average and collect rewards.

    Returns:
    - List of average rewards collected at each interval.
    """
    rewards = []
    for i in range(0, num_episodes, interval):
        avg_reward = evaluate_policy(env, Q, interval)
        rewards.append(avg_reward)
    return rewards

def main():
    env = gym.make("FrozenLake-v1")
    num_episodes = 10000
    
    ## Train with fixed learning rate and collect rewards
    print("Training with fixed learning rate...")
    Q_fixed = q_learning(env, num_episodes, learning_rate=0.2, discount_factor=0.99, epsilon=1.0, dynamic_lr=False)
    fixed_lr_rewards = collect_rewards(env, Q_fixed, num_episodes)
    overall_avg_reward_fixed = np.mean(fixed_lr_rewards)
    print(f"Overall average reward with fixed learning rate: {overall_avg_reward_fixed:.2f}")
    
    # Train with dynamic learning rate and collect rewards
    print("Training with dynamic learning rate...")
    Q_dynamic = q_learning(env, num_episodes, learning_rate=0.2, discount_factor=0.99, epsilon=1.0, dynamic_lr=True)
    dynamic_lr_rewards = collect_rewards(env, Q_dynamic, num_episodes)
    overall_avg_reward_dynamic = np.mean(dynamic_lr_rewards)
    print(f"Overall average reward with dynamic learning rate: {overall_avg_reward_dynamic:.2f}")
    
    # Plot the collected rewards for comparison
    plot_results(fixed_lr_rewards, dynamic_lr_rewards, num_episodes)


    # # Demonstrate agent's performance with fixed learning rate policy
    # visual_env_fixed = gym.make('FrozenLake-v1', render_mode='human')
    # print("Demonstrating agent's performance with fixed learning rate policy...")
    # demo_agent(visual_env_fixed, Q_fixed, 'fixed', 3)

    # # Demonstrate agent's performance with dynamic learning rate policy
    # visual_env_dynamic = gym.make('FrozenLake-v1', render_mode='human')
    # print("Demonstrating agent's performance with dynamic learning rate policy...")
    # demo_agent(visual_env_dynamic, Q_dynamic, 'dynamic', 3)

if __name__ == '__main__':
    main()
