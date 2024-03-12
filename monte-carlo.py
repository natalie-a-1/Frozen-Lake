import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class MonteCarlo:
    def __init__(self, env, gamma=0.9, max_episodes=50000, epsilon=1.0, decay_rate=0.001, min_epsilon=0.01, importance_sampling=False):
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
        self.average_rewards = []

    def train(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()[0]
            done = False
            score = 0

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state, :])
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if self.importance_sampling:
                    weight = self.calculate_importance_sampling_weight(state, action)

                else:
                    weight = 1

                self.q_table[state, action] += weight * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action])
                done = terminated or truncated
                score += reward
                state = next_state

            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate * self.epsilon)
            self.reward_list.append(score)

            if episode % 1000 == 0:
                average_reward = np.mean(self.reward_list[-1000:])
                self.average_rewards.append(average_reward)

        if self.importance_sampling:       
            print(f"Final score of MC with importance sampling: {sum(self.reward_list) / self.max_episodes}")
        else:
            print(f"Final score of standard MC: {sum(self.reward_list) / self.max_episodes}")
        return self.average_rewards
    
    def calculate_importance_sampling_weight(self, state, action):
        n_actions = self.action_dim 

        if action == np.argmax(self.q_table[state, :]):
            target_policy_prob = 1 - self.epsilon + (self.epsilon / n_actions)
        else:
            target_policy_prob = self.epsilon / n_actions

        behavior_policy_prob = 1 - self.epsilon + (self.epsilon / n_actions) if action == np.argmax(self.q_table[state, :]) else self.epsilon / n_actions
        weight = target_policy_prob / behavior_policy_prob

        return weight

def plot_average_rewards(standard_rewards, is_rewards):
    plt.figure(figsize=(10, 5))
    episodes = np.arange(len(standard_rewards))
    plt.plot(episodes, standard_rewards, label='Standard MC')
    plt.plot(episodes, is_rewards, label='MC with Importance Sampling')
    plt.title("Average Reward Over Time")
    plt.xlabel("Episode (x100)")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
standard_mc_agent = MonteCarlo(env, importance_sampling=False)
is_mc_agent = MonteCarlo(env, importance_sampling=True)

standard_rewards = standard_mc_agent.train()
is_rewards = is_mc_agent.train()

plot_average_rewards(standard_rewards, is_rewards)
