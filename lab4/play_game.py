from Climber import Climber
import pygame
import numpy as np


def play_climber():
    env = Climber(render_mode='human', rock_every=100, target_size=20, rock_size=30, gravity=495,
                  swing_power=650, pickaxe_length=60, rock_mass=0.75)
    obs, info = env.reset()
    env.render()
    pickaxe_active = False
    running = True

    print("Sterowanie:")
    print("- SPACJA: Wbij/Wyjmij kilof (Przełącznik)")
    print("- STRZAŁKA W LEWO: Siła w lewo")
    print("- STRZAŁKA W PRAWO: Siła w prawo")
    print("- ZAMKNIĘCIE OKNA: Wyjście")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pickaxe_active = not pickaxe_active

        keys = pygame.key.get_pressed()
        force = 0.0
        if keys[pygame.K_LEFT]:
            force = -1.0
        if keys[pygame.K_RIGHT]:
            force = 1.0

        pickaxe_val = 1.0 if pickaxe_active else -1.0

        action = np.array([pickaxe_val, force], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            status = "WYGRANA!" if info.get("is_success") else "PORAŻKA (Kamień)"
            if truncated:
                status = "KONIEC CZASU"

            print(f"Koniec epizodu! Status: {status} | Ostatnia nagroda: {reward:.2f}")

            obs, info = env.reset()
            pickaxe_active = False

    env.close()

if __name__ == "__main__":
    play_climber()