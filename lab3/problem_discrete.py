import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate(game_name, * , alpha, discount_level, eps, eps_decay, episodes, **kwargs):
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

if __name__ == "__main__":
    alpha = 0.5
    discount_level = 0.9
    epsilon = 1
    eps_decay = 0.001
    episodes = 1500
    simulate('FrozenLake-v1', alpha=alpha, discount_level=discount_level,
             eps=epsilon, eps_decay=eps_decay, episodes=episodes, is_slippery= False, render_mode = None)