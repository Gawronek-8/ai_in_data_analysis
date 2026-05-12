from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import time
from Climber import Climber
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


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

    model = PPO("MlpPolicy", env, verbose=0, device=device, tensorboard_log="./ppo_climber_tensorboard/")

    logger_callback = EpisodeLoggerCallback()

    model.learn(total_timesteps=10000, callback=logger_callback, tb_log_name="PPO_run_1")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Średnia nagroda: {mean_reward}")

    model.save("ppo_climber_model")
    torch.save(model.policy.state_dict(), "climber_weights_only.pth")

    time.sleep(2)

    print("Model zapisany pomyślnie. Możesz zamknąć program.")


def test_train_with_curriculum():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_env(rock_val):
        env = Climber(render_mode=None, rock_every=rock_val)
        return Monitor(env)

    print("--- Faza 1: Nauka podstaw ---")
    venv = DummyVecEnv([lambda: make_env(1000)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=0,
        tensorboard_log="./ppo_climber_tensorboard/",
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=128,
        n_epochs=15,
        gamma=0.98,
        ent_coef=0.1,
        clip_range=0.1,
        device=device
    )
    model.learn(total_timesteps=150000, tb_log_name="phase_1")

    print("--- Faza 2: Średnie zagęszczenie ---")
    venv.venv = DummyVecEnv([lambda: make_env(300)])
    venv.reset()
    model.set_env(venv)
    model.learn(total_timesteps=150000, tb_log_name="phase_2", reset_num_timesteps=False)

    print("--- Faza 3: Trudne warunki ---")
    venv.venv = DummyVecEnv([lambda: make_env(110)])
    venv.reset()
    model.set_env(venv)
    model.learn(total_timesteps=200000, tb_log_name="phase_3", reset_num_timesteps=False)

    model.save("ppo_climber_model")
    venv.save("vec_normalize.pkl")

    torch.save(model.policy.state_dict(), "climber_weights_only.pth")
    print("Model i statystyki zapisane.")


if __name__ == "__main__":
    test_train_with_curriculum()