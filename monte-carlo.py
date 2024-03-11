import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class MonteCarlo:
    def __init__(self, env, gamma=0.9, max_episodes=100000, epsilon=1.0, decay_rate=0.0001, min_epsilon=0.001):
        self.env = env
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.q_table = np.zeros((self.state_dim, self.action_dim))
        self.new_q = np.zeros((self.state_dim, self.action_dim))
        self.reward_list = []
        self.average_rewards = []  # To store average rewards

    def train(self):
        for episode in range(self.max_episodes):
            state_info = env.reset()
            state = state_info[0]
            done = False
            episode_rewards = []
            episode_states_actions = []
            score = 0

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state, :])
                
                result = env.step(action)
                new_state, reward, terminated, truncated, _ = result if len(result) == 5 else result[0], result[1], result[2], result[3], {}

                episode_states_actions.append((state, action))
                done = terminated or truncated
                score += reward
                state = new_state[0]
            
            self.reward_list.append(score)

            for state, action in episode_states_actions:
                self.new_q[int(state), action] += 1
                learning_rate = 1.0 / self.new_q[state, action]
                self.q_table[state, action] += learning_rate * (score - self.q_table[state, action])

            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay_rate * episode))

            if episode % 1000 == 0:
                average_reward = sum(self.reward_list[-1000:]) / 1000
                self.average_rewards.append(average_reward)
                print(f"Episode {episode}/{self.max_episodes} - Average Reward: {average_reward}")
        
        print(f"Final score: {sum(self.reward_list) / self.max_episodes}")
        self.plot_average_rewards()

    def plot_average_rewards(self):
        episodes_per_point = 1000  # This should match the frequency of your averaging
        episodes = np.arange(len(self.average_rewards)) * episodes_per_point

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, self.average_rewards, label='Average Reward')
        plt.title("Average Reward Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")

        # Improve readability of the x-axis by adjusting the number of ticks
        max_episode = len(self.average_rewards) * episodes_per_point
        plt.xticks(np.arange(0, max_episode, step=max_episode / 10))  # Adjust step for more or fewer ticks

        plt.legend()
        plt.grid(True)  # Optional: Adds a grid for easier reading
        plt.show()
# Example usage
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
mc_agent = MonteCarlo(env)
mc_agent.train()
