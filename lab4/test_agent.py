from stable_baselines3 import PPO
from Climber import Climber
import pygame

def test_agent():
    env = Climber(render_mode='human', rock_every=20)
    model = PPO.load("ppo_climber_model")
    obs, info = env.reset()

    env.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == '__main__':
    test_agent()