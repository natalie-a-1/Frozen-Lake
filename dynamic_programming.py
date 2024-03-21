import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make('FrozenLake-v1', is_slippery=False)

def policy_evaluation(env, policy, gamma=0.9, theta=1e-4):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(env, V, policy, gamma=1):
    policy_stable = True
    new_policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        old_action = np.argmax(policy[s])
        q = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                q[a] += prob * (reward + gamma * V[next_state])
        new_action = np.argmax(q)
        new_policy[s, new_action] = 1.0
        if old_action != new_action:
            policy_stable = False
    return new_policy, policy_stable

def policy_iteration(env, gamma=1, theta=1e-4, max_iterations=1000):
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    avg_values_over_time = []
    for iteration in range(max_iterations):
        V = policy_evaluation(env, policy, gamma, theta)
        policy, policy_stable = policy_improvement(env, V, policy, gamma)
        avg_values_over_time.append(np.mean(V))
        if policy_stable:
            print(f"Policy stabilized after {iteration+1} iterations.")
            break
    return policy, V, avg_values_over_time

policy_pi, V_pi, avg_values = policy_iteration(env)

# Plot the learning curve
plt.plot(avg_values, marker='o')
plt.title('Learning Curve: Average Value Function Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Average Value')
plt.grid(True)
plt.show()

def evaluate_policy(env, policy, num_episodes=100):
    total_rewards = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(policy[state])
            next_state, reward, done, info = env.step(action)
            total_rewards += reward
    average_reward = total_rewards / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")

evaluate_policy(env, policy_pi, num_episodes=100)
