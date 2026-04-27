import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def train_car_racing(discount_val):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("CarRacing-v3", continuous=True)
    env = Monitor(env)

    steps = 2048 * 50
    model = PPO("CnnPolicy", env, verbose=0, gamma=discount_val, learning_rate=0.0003, device=device)

    model.learn(total_timesteps=steps, progress_bar=True)

    model.save(ROOT_DIR / f"model{discount_val}")

    episode_rewards = env.get_episode_rewards()

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Współczynnik dyskontowy: {discount_val} -> Średnia nagroda: {mean_reward}")

    return episode_rewards


if __name__ == "__main__":
    discount_vals = [0.1, 0.9, 0.999]
    all_results = {}

    plt.figure(figsize=(10, 6))

    for g in discount_vals:
        rewards = train_car_racing(g)
        all_results[g] = rewards

        if len(rewards) > 10:
            window = min(10, len(rewards) // 2)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            plt.plot(smoothed, label=f"Współczynnik dyskontowy {g}")
        else:
            plt.plot(rewards, label=f"Współczynnik dyskontowy {g} (bez okna)")

    plt.title("Porównanie uczenia CarRacing dla różnych wartości dyskontowych")
    plt.xlabel("Epizody")
    plt.ylabel("Nagroda")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig("continuous_problem.png", dpi=300, bbox_inches='tight')
    plt.show()
