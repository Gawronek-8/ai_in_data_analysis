import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def test_agent(model_path):
    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

    model = PPO.load(model_path)

    obs, _ = env.reset()
    total_reward = 0


    for _ in range(2000):  
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            total_reward = 0

    env.close()


if __name__ == "__main__":
    test_agent(ROOT_DIR / "model0.9")