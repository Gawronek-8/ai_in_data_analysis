from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from Climber import Climber
import torch


class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.recent_rewards = deque(maxlen=10)
        self.episode_count = 0

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            ep_length = self.locals["infos"][0]["episode"]["l"]
            self.recent_rewards.append(ep_reward)
            self.episode_count += 1
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)

            print(f"Epizod: {self.episode_count} | Kroki: {ep_length} | Nagroda: {ep_reward:.2f} | Średnia (ostatnie 10): {avg_reward:.2f}")

        return True



def test_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = Climber(render_mode=None, rock_every=150)
    env = Monitor(env)

    model = PPO("MlpPolicy", env, verbose=0, device=device)

    logger_callback = EpisodeLoggerCallback()

    model.learn(total_timesteps=350_000, callback=logger_callback)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Średnia nagroda: {mean_reward}")

    model.save("ppo_climber_model")


if __name__ == "__main__":

    test_ppo()