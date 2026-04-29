import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_q_learning(game_name, * , alpha, discount_level, eps, eps_decay, episodes, **kwargs):
    env = gym.make(game_name, **kwargs)

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    rewards_per_episode = []

    for _ in tqdm(range(episodes)):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            q_table[state, action] = old_value + alpha * (reward + discount_level * next_max - old_value)

            state = next_state
            total_reward += reward

        eps = max(eps - eps_decay, 0.01)
        rewards_per_episode.append(total_reward)

    plt.plot(np.convolve(rewards_per_episode, np.ones(100) / 100, mode='valid'))
    plt.title(f"Krzywa uczenia - Frozen Lake (Współczynnik dyskontowy={discount_level})")
    plt.xlabel("Epizody")
    plt.ylabel("Nagroda (średnia z 100)")
    plt.show()

def simulate_value_iteration(game_name, discount_level=0.9, theta=1e-6, **kwargs):
    env = gym.make(game_name, **kwargs)
    env = env.unwrapped

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    deltas = []
    mean_values = []

    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]

            q_values = []
            for a in range(n_actions):
                q = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    q += prob * (reward + discount_level * V[next_state])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        deltas.append(delta)
        mean_values.append(np.mean(V))

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q += prob * (reward + discount_level * V[next_state])
            q_values.append(q)

        policy[s] = np.argmax(q_values)

    plt.plot(deltas)
    plt.title("Zbieżność Value Iteration (delta)")
    plt.xlabel("Iteracje")
    plt.ylabel("Max zmiana V")
    plt.show()

    plt.plot(mean_values)
    plt.title("Średnia wartość stanów")
    plt.xlabel("Iteracje")
    plt.ylabel("Mean V(s)")
    plt.show()

    return V, policy

def evaluate_policy_value_iteration(env, policy, episodes=100):
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return rewards

def simulate_monte_carlo_control(game_name, episodes, discount_level=0.9, eps=1.0, eps_decay=0.001, **kwargs):
    env = gym.make(game_name, **kwargs)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    rewards_per_episode = []

    for _ in tqdm(range(episodes)):
        state, _ = env.reset()

        episode = []
        terminated = truncated = False

        while not (terminated or truncated):
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))

            state = next_state

        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = discount_level * G + r

            if (s, a) not in visited:
                visited.add((s, a))

                returns_count[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / returns_count[s, a]

        eps = max(eps - eps_decay, 0.01)
        rewards_per_episode.append(sum([x[2] for x in episode]))

    plt.plot(np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid'))
    plt.title("Monte Carlo Control - Frozen Lake")
    plt.show()

    return Q

if __name__ == "__main__":
    alpha = 0.5
    discount_level = 0.99
    epsilon = 0.5
    eps_decay = 0.0005
    episodes = 1500

    alg = 3

    if alg == 1:
        # ALG1 - Q-Learning
        simulate_q_learning('FrozenLake-v1', alpha=alpha, discount_level=discount_level,
                 eps=epsilon, eps_decay=eps_decay, episodes=episodes, is_slippery=False, render_mode=None)
    elif alg == 2:
        # ALG2 - Value Iteration
        V, policy = simulate_value_iteration('FrozenLake-v1', discount_level=discount_level, is_slippery=True)
        print(policy.reshape(4, 4))
        env = gym.make('FrozenLake-v1', is_slippery=True)
        rewards = evaluate_policy_value_iteration(env, policy, episodes=1000)

        plt.plot(np.convolve(rewards, np.ones(100) / 100, mode='valid'))
        plt.title("Reward dla policy z Value Iteration")
        plt.show()
    elif alg == 3:
        # ALG3 - Monte Carlo Control
        simulate_monte_carlo_control('FrozenLake-v1', episodes=episodes, discount_level=discount_level,eps=epsilon, eps_decay=eps_decay, is_slippery=True)