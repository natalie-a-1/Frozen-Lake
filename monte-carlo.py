# import gym
# import numpy as np
# import random

# class MonteCarloAgent:
#     def __init__(self, gamma=0.9, max_episodes=10000, epsilon=0.1):
#         self.gamma = gamma
#         self.max_episodes = max_episodes
#         self.epsilon = epsilon  # Exploration rate

#         self.state_dim = env.observation_space.n
#         self.action_dim = env.action_space.n

#         self.q_table = np.zeros((self.state_dim, self.action_dim))
#         self.returns = {(s, a): [] for s in range(self.state_dim) for a in range(self.action_dim)}
#         self.policy = np.random.choice(self.action_dim, self.state_dim)

#     def generate_episode(self):
#         episode = []
#         state_info = env.reset()
#         state = state_info[0] if isinstance(state_info, tuple) else state_info
#         done = False

#         while not done:
#             action = self.get_action(state)
#             result = env.step(action)
#             next_state, reward, terminated, truncated, _ = result if len(result) == 5 else result[0], result[1], result[2], result[3], {}
#             next_state = next_state[0] if isinstance(next_state, tuple) else next_state
#             episode.append((state, action, reward))
#             state = next_state

#         return episode

#     def get_action(self, state):
#         # Epsilon-greedy policy
#         if random.uniform(0, 1) < self.epsilon:
#             return env.action_space.sample()
#         else:
#             return np.argmax(self.q_table[state, :])

#     def update_q_table(self, episode):
#         G = 0
#         for state, action, reward in reversed(episode):
#             G = self.gamma * G + reward
#             if not (state, action) in [(x[0], x[1]) for x in episode[:-1]]:
#                 self.returns[(state, action)].append(G)
#                 self.q_table[state][action] = np.mean(self.returns[(state, action)])

#     def derive_policy(self):
#         for state in range(self.state_dim):
#             self.policy[state] = np.argmax(self.q_table[state])

#     def train(self):
#         for episode in range(1, self.max_episodes + 1):
#             episode_data = self.generate_episode()
#             self.update_q_table(episode_data)
#             self.derive_policy()

#             if episode % 1000 == 0:
#                 print(f"Episode {episode}/{self.max_episodes} completed.")

# # Example usage
# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
# mc_agent = MonteCarloAgent(gamma=0.9, max_episodes=10000, epsilon=0.1)
# mc_agent.train()

# print("Derived policy:")
# print(mc_agent.policy)
print("hello")