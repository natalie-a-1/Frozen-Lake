import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class MonteCarlo:
    def __init__(self, env, gamma=0.8, max_episodes=15000, epsilon=0.9, decay_rate=0.001, min_epsilon=0.01, importance_sampling=False):
        self.env = env
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.importance_sampling = importance_sampling  # Flag for importance sampling
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        self.q_table = np.zeros((self.state_dim, self.action_dim))
        self.reward_list = []
        self.episode_rewards = []
        self.q_table = np.zeros((self.state_dim, self.action_dim))
        self.returns_sum = np.zeros((self.state_dim, self.action_dim))
        self.returns_count = np.zeros((self.state_dim, self.action_dim))

    def train(self):
        episode_rewards = []

        for episode in range(self.max_episodes):
            episode_sequence = []
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state, :])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_sequence.append((state, action, reward))
                done = terminated or truncated
                total_reward += reward
                state = next_state

            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate * self.epsilon)
            episode_rewards.append(total_reward)

            #self.q_table[state, action] += weight * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action])
            G = 0
            W = 1.0
            for state, action, reward in reversed(episode_sequence):
                G = self.gamma * G + reward
                self.returns_sum[state, action] += W * G
                self.returns_count[state, action] += W
                self.q_table[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
                
                if self.importance_sampling:
                    target_policy = np.argmax(self.q_table[state, :])
                    W *= self.calculate_importance_sampling_weight(state, action, target_policy)
                    if W == 0:
                        break

        if self.importance_sampling:       
            print(f"Final score of MC with importance sampling: {sum(episode_rewards)/len(episode_rewards)}")
        else:
            print(f"Final score of standard MC: {sum(episode_rewards)/len(episode_rewards)}")

        return episode_rewards

    def calculate_importance_sampling_weight(self, state, action, target_policy):
        # Assuming target policy is greedy regarding the current Q-table
        behavior_policy_prob = self.epsilon / self.action_dim + (1 - self.epsilon if action == target_policy else 0)
        target_policy_prob = 1.0 if action == target_policy else 0
        return target_policy_prob / behavior_policy_prob if behavior_policy_prob > 0 else 0


def plot_learning_curves(mean_rewards_standard, mean_rewards_is, window_size=100):
    plt.figure(figsize=(15, 7))

    def moving_average(rewards, window_size):
        if len(rewards) < window_size:
            return rewards
        moving_averages = []
        for i in range(len(rewards) - window_size + 1):
            window_average = sum(rewards[i:i + window_size]) / window_size
            moving_averages.append(window_average)

        return moving_averages

    smoothed_standard = moving_average(mean_rewards_standard, window_size)
    smoothed_is = moving_average(mean_rewards_is, window_size)

    episodes_standard = range(len(smoothed_standard))
    episodes_is = range(len(smoothed_is))

    if smoothed_standard:
        plt.plot(episodes_standard, smoothed_standard, label='Standard MC', alpha=0.75)
    if smoothed_is:
        plt.plot(episodes_is, smoothed_is, label='MC with Importance Sampling', alpha=0.75)

    plt.title("Monte Carlo: Average Rewards vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_multiple_trials(env_name, trials=30, importance_sampling=False):
    all_episode_rewards = []

    for trial in range(trials):
        env = gym.make(env_name, desc=None, map_name="4x4", is_slippery=False)
        agent = MonteCarlo(env, importance_sampling=importance_sampling)
        episode_rewards = agent.train()
        all_episode_rewards.append(episode_rewards)

    mean_rewards_per_episode = np.mean(all_episode_rewards, axis=0)
    std_reward = np.std(all_episode_rewards)
    return mean_rewards_per_episode, std_reward, all_episode_rewards 

env_name = 'FrozenLake-v1'
trials = 30

mean_rewards_standard, std_standard, total_standard = run_multiple_trials(env_name, trials, importance_sampling=False)
mean_rewards_is, std_is, total_is = run_multiple_trials(env_name, trials, importance_sampling=True)

plot_learning_curves(mean_rewards_standard, mean_rewards_is, window_size=100)

print(f"Standard MC: STD = {std_standard}")
print(f"Mean: {np.mean(mean_rewards_standard)}")
print(f"MC with Importance Sampling: STD = {std_is}")
print(f"Mean: {np.mean(mean_rewards_is)}")

