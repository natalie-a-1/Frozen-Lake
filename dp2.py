import numpy as np
import gym
import matplotlib.pyplot as plt
import time

def policy_evaluation(env, policy, gamma, threshold):
    num_states = env.observation_space.n
    V = np.zeros(num_states)
    deltas = []  # Track max difference in value function per iteration
    while True:
        delta = 0
        for state in range(num_states):
            v = V[state]
            action = policy[state]
            V[state] = sum([proba * (reward + gamma * V[new_state]) for proba, new_state, reward, _ in env.P[state][action]])
            delta = max(delta, abs(v - V[state]))
        deltas.append(delta)
        if delta < threshold:
            break
    return V, deltas

def policy_evaluation_with_prioritization(env, policy, gamma, threshold):
    num_states = env.observation_space.n
    V = np.zeros(num_states)
    deltas = []  
    priority_queue = {state: 0 for state in range(num_states)}  # State priority initialization

    while priority_queue:
        state, _ = max(priority_queue.items(), key=lambda x: x[1])  # State with highest priority
        priority_queue.pop(state)  # Remove this state from priority queue
        
        v = V[state]
        action = policy[state]
        V[state] = sum([proba * (reward + gamma * V[new_state]) for proba, new_state, reward, _ in env.P[state][action]])
        delta = abs(v - V[state])
        deltas.append(delta)

        if delta > threshold:
            for action in range(env.action_space.n):
                for proba, new_state, reward, _ in env.P[state][action]:
                    if new_state not in priority_queue:
                        priority_value = abs(reward + gamma * V[new_state] - V[state])
                        priority_queue[new_state] = priority_value

    return V, deltas

def policy_improvement(env, V, gamma):
    num_states = env.observation_space.n
    policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for proba, new_state, reward, _ in env.P[state][action]:
                action_values[action] += proba * (reward + gamma * V[new_state])
        best_action = np.argmax(action_values)
        policy[state] = best_action
    return policy

def policy_iteration(env, gamma=0.99, threshold=0.0001, prioritized=False):
    num_states = env.observation_space.n
    policy = np.random.randint(low=0, high=env.action_space.n, size=(num_states,))
    all_deltas = []  # List to accumulate deltas from all policy evaluations
    evaluation_function = policy_evaluation_with_prioritization if prioritized else policy_evaluation
    
    while True:
        start_time = time.time()
        V, deltas = evaluation_function(env, policy, gamma, threshold)
        all_deltas.extend(deltas)
        new_policy = policy_improvement(env, V, gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, V, all_deltas, time.time() - start_time

def plot_comparative_learning_curves(deltas_standard, deltas_prioritized, title='Comparison of Policy Iteration Convergence'):
    plt.figure(figsize=(12, 8))
    plt.plot(deltas_standard, label='Standard Policy Iteration', color='blue')
    plt.plot(deltas_prioritized, label='Prioritized Policy Iteration', color='red')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Max Value Function Change (Delta)')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_episodes_for_stats(environment, policy, n_episodes=1000):
    rewards = []
    for _ in range(n_episodes):
        state_info = environment.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        episode_reward = 0
        done = False
        while not done:
            action = policy[state]
            next_state, reward, terminated, truncated, _= environment.step(action)
            episode_reward += reward
            state = next_state
            done =  terminated or truncated
        rewards.append(episode_reward)
    return rewards

def calculate_statistics(rewards):
    average_reward = np.mean(rewards)
    std_dev_reward = np.std(rewards)
    success_rate = sum(r > 0 for r in rewards) / len(rewards)
    return average_reward, std_dev_reward, success_rate

# Main
env = gym.make('FrozenLake-v1', is_slippery=True)
optimal_policy_standard, _, deltas_standard, time_standard = policy_iteration(env)
optimal_policy_prioritized, _, deltas_prioritized, time_prioritized = policy_iteration(env, prioritized=True)

# Gather statistics
rewards_standard = run_episodes_for_stats(env, optimal_policy_standard)
rewards_prioritized = run_episodes_for_stats(env, optimal_policy_prioritized)

avg_reward_standard, std_reward_standard, success_rate_standard = calculate_statistics(rewards_standard)
avg_reward_prioritized, std_reward_prioritized, success_rate_prioritized = calculate_statistics(rewards_prioritized)

# Print statistics
print(f"Standard Policy Iteration Time: {time_standard:.4f}s")
print(f"Prioritized Policy Iteration Time: {time_prioritized:.4f}s")
print("\nStandard Policy Iteration Performance:")
print(f"Average Reward: {avg_reward_standard:.4f}, Standard Deviation: {std_reward_standard:.4f}, Success Rate: {success_rate_standard:.2%}")
print("\nPrioritized Policy Iteration Performance:")
print(f"Average Reward: {avg_reward_prioritized:.4f}, Standard Deviation: {std_reward_prioritized:.4f}, Success Rate: {success_rate_prioritized:.2%}")

# Comparative learning curves
plot_comparative_learning_curves(deltas_standard, deltas_prioritized)
