import gymnasium as gym
import numpy as np
from problem_discrete import simulate_value_iteration, evaluate_policy_value_iteration

def evaluate_q_learning(params):
    alpha, gamma, eps, eps_decay = params

    env = gym.make('FrozenLake-v1', is_slippery=False)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    total_discounted_reward = 0

    for episode in range(1000):
        state, _ = env.reset()
        terminated = truncated = False
        G = 0
        t = 0

        while not (terminated or truncated):
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update
            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            G += (gamma ** t) * reward
            t += 1
            state = next_state

        eps = max(eps - eps_decay, 0.01)
        total_discounted_reward += G

    return total_discounted_reward

def evaluate_mc(params):
    gamma, eps, eps_decay = params

    env = gym.make('FrozenLake-v1', is_slippery=True)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns_count = np.zeros_like(Q)

    total_reward = 0

    for episode in range(1000):
        state, _ = env.reset()
        episode_data = []

        terminated = truncated = False

        while not (terminated or truncated):
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()

        for t in reversed(range(len(episode_data))):
            s, a, r = episode_data[t]
            G = gamma * G + r

            if (s, a) not in visited:
                visited.add((s, a))
                returns_count[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / returns_count[s, a]

        total_reward += G
        eps = max(eps - eps_decay, 0.01)

    return total_reward

def evaluate_vi(gamma):
    V, policy = simulate_value_iteration('FrozenLake-v1', discount_level=gamma, is_slippery=True)

    env = gym.make('FrozenLake-v1', is_slippery=True)

    rewards = evaluate_policy_value_iteration(env, policy, episodes=1000)

    return sum(rewards)


alg = 3

if alg == 1:
    alphas = [0.1, 0.3, 0.5]
    gammas = [0.8, 0.9, 0.99]
    epsilons = [1.0, 0.5]
    decays = [0.001, 0.0005]

    best_score = -np.inf
    best_params = None

    for a in alphas:
        for g in gammas:
            for e in epsilons:
                for d in decays:
                    score = evaluate_q_learning((a, g, e, d))
                    if score > best_score:
                        best_score = score
                        best_params = (a, g, e, d)

    print(best_params, best_score)
elif alg == 2:
    gammas = [0.5, 0.7, 0.9, 0.99]

    for g in gammas:
        score = evaluate_vi(g)
        print(g, score)
elif alg == 3:
    gammas = [0.8, 0.9, 0.99]
    epsilons = [1.0, 0.5]
    decays = [0.001, 0.0005]

    best_score = -np.inf
    best_params = None

    for g in gammas:
        for e in epsilons:
            for d in decays:
                score = evaluate_mc((g, e, d))
                if score > best_score:
                    best_score = score
                    best_params = (g, e, d)

    print(best_params, best_score)