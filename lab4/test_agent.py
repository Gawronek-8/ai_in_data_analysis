from stable_baselines3 import PPO
from Climber import Climber
import pygame
import torch
import zipfile
import io
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def test_agent():
    # env = Climber(render_mode='human', rock_every=110)
    # 1. Tworzymy bazowe środowisko
    env = Climber(render_mode='human', rock_every=110)
    # 2. Musimy je owinąć tak samo jak w treningu
    venv = DummyVecEnv([lambda: env])

    try:
        # 3. WCZYTUJEMY STATYSTYKI NORMALIZACJI
        venv = VecNormalize.load("vec_normalize.pkl", venv)
        # BARDZO WAŻNE: Wyłączamy naukę statystyk i normalizację nagród podczas testu
        venv.training = False
        venv.norm_reward = False
    except Exception as e:
        print(f"Błąd ładowania: {e}")
        # Tutaj Twoja stara logika wstrzykiwania wag (opcjonalnie)
        return

    try:
        model = PPO.load("ppo_climber_model", env=venv)
    except:
        model = PPO("MlpPolicy", venv, verbose=1)

        model_path = "ppo_climber_model.zip"

        try:
            with zipfile.ZipFile(model_path, 'r') as archive:
                with archive.open('policy.pth') as weights_file:
                    buffer = io.BytesIO(weights_file.read())
                    state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

                    model.policy.load_state_dict(state_dict)
                    print("Wagi wstrzyknięte pomyślnie!")
        except Exception as e:
            print(f"Błąd podczas ręcznego ładowania: {e}")
            return

    obs = venv.reset()
    env.render()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = venv.step(action)

        env.render()

        # if dones[0]:
        #     obs = venv.reset()

    env.close()

if __name__ == '__main__':
    test_agent()