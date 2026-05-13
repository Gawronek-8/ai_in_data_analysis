import math
from typing import Any

import numpy as np
import pygame
import pymunk
from pymunk import pygame_util
from gymnasium import Env, spaces
from gymnasium.core import ObsType
from object_categories import ObjectCategory


def _get_relative_pos(body1: pymunk.Body, body2: pymunk.Body) -> tuple[float, float]:
    return  body1.position.x - body2.position.x, body1.position.y - body2.position.y


class Climber(Env):
    """
    Climber game environment for gymnasium. Observation space is continuous, same as action space.
    Number of rays used in ray-casting is hardcoded and is 12. Utilizes pymunk as the physics engine.
    """


    metadata = {
        'render_modes': ['human'],
        'fps' : 60,
        'map_settings' : {
            'width': 800,
            'height': 800,
        }
    }



    def __init__(self, render_mode = None, rock_every: float = 0, player_size = 15, rock_size = 15,
                 target_size = 15, rock_mass = 0.75, gravity = 450, pickaxe_length = 50, swing_power = 650):

        self.target_body = None
        self.player_body = None
        self.draw_options = None
        self.clock = None
        self.window = None
        self.space = None
        self.walls = None
        self.target = None
        self.player = None
        self.joint = None
        self.prev_pos = None

        assert rock_every > 0

        assert render_mode in self.metadata['render_modes'] or render_mode is None

        self.render_mode = render_mode
        self.rock_every = rock_every

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)

        self.hit_rock = False
        self.hit_target = False
        self.curr_step = 0
        self.max_steps = 10_000
        self.pickaxe_length = pickaxe_length
        self.gravity_level = gravity
        self.swing_power_multiplier = swing_power
        self.target_size = target_size
        self.player_size = player_size
        self.rock_size = rock_size
        self.rock_mass = rock_mass
        self.ray_distance = 200


    def step(self, action):

        pickaxe_val = action[0]
        player_force = action[1]

        self.prev_pos = self.player.body.position

        if pickaxe_val > 0.5 and self.joint is None:
            self.joint = self._create_pickaxe()
            self.space.add(self.joint)
        elif pickaxe_val < -0.5 and self.joint:
            self.space.remove(self.joint)
            self.joint = None

        force_x = player_force * self.swing_power_multiplier
        if self.joint is not None:
            self.player.body.apply_force_at_local_point((force_x, 0), (0, 0))
        else:
            self.player.body.apply_force_at_local_point((force_x * 0.2, 0), (0, 0))

        game_fps = self.metadata['fps']

        self.space.step(1 / game_fps)
        self.curr_step += 1
        obs = self._get_obs()

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        reward -= 0.001

        if self.hit_rock:
            reward -= 2.0
            terminated = True
            info["is_success"] = False

        elif self.hit_target:
            reward += 20.0
            terminated = True

        elif self.curr_step >= self.max_steps:
            truncated = True

        elif self.hit_target:
            reward += 300
            terminated = True
            info["is_success"] = True

        else:
            prev_dist = np.linalg.norm(self.prev_pos - self.target.body.position)
            curr_dist = np.linalg.norm(self.player.body.position - self.target.body.position)

            reward += (prev_dist - curr_dist) * 0.1
            reward += (self.prev_pos.y - self.player.body.position.y) * 0.05

            if self.joint is not None:
                reward += 0.01

            reward += (self.player.body.position.y / 1000) * 0.01
            reward -= abs(self.player.body.angular_velocity) * 0.05

        if self.curr_step % self.rock_every == 0:
            rock, rock_body = self._create_rock()
            self.space.add(rock_body)
            self.space.add(rock)

        return obs, reward, terminated, truncated, info


    def render(self):
        if self.render_mode != 'human':
            return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            width = self.metadata['map_settings']['width']
            height = self.metadata['map_settings']['height']
            self.window = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()

            self.player_img_base = pygame.image.load("character_green_walk_a.png").convert_alpha()
            self.player_img_base = pygame.transform.scale(self.player_img_base,
                                                          (self.player_size * 2, self.player_size * 2))

            self.rock_img_base = pygame.image.load("block_fall.png").convert_alpha()
            self.rock_img_base = pygame.transform.scale(self.rock_img_base, (self.rock_size * 2, self.rock_size * 2))

            self.target_img_base = pygame.image.load("ladybug_fly.png").convert_alpha()
            self.target_img_base = pygame.transform.scale(self.target_img_base,
                                                          (self.target_size * 2, self.target_size * 2))

        self.window.fill((255, 255, 255))



        for wall in self.walls:
            p1 = (int(wall.a.x), int(wall.a.y))
            p2 = (int(wall.b.x), int(wall.b.y))
            pygame.draw.line(self.window, (50, 50, 50), p1, p2, int(wall.radius * 2))

        if self.joint is not None:
            p1 = (int(self.joint.anchor_a.x), int(self.joint.anchor_a.y))
            p2 = (int(self.player_body.position.x), int(self.player_body.position.y))
            pygame.draw.line(self.window, (139, 69, 19), p1, p2, 4)

        p_angle = math.degrees(-self.player_body.angle)
        p_img_rotated = pygame.transform.rotate(self.player_img_base, p_angle)
        p_rect = p_img_rotated.get_rect(center=(self.player_body.position.x, self.player_body.position.y))
        self.window.blit(p_img_rotated, p_rect.topleft)

        t_rect = self.target_img_base.get_rect(center=(self.target_body.position.x, self.target_body.position.y))
        self.window.blit(self.target_img_base, t_rect.topleft)

        for shape in self.space.shapes:
            if shape.collision_type == ObjectCategory.ROCK:
                r_angle = math.degrees(-shape.body.angle)
                r_img_rotated = pygame.transform.rotate(self.rock_img_base, r_angle)
                r_rect = r_img_rotated.get_rect(center=(shape.body.position.x, shape.body.position.y))
                self.window.blit(r_img_rotated, r_rect.topleft)

        pygame.display.flip()
        self.clock.tick(self.metadata['fps'])



    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.space = pymunk.Space()

        self.target, self.target_body = self._create_target()
        self.player, self.player_body = self._create_player()
        self.prev_pos = self.player.body.position
        self.walls = self._create_walls()
        self.joint = None

        self.space.add(self.player, self.player_body, self.target, self.target_body, *self.walls)
        self.space.gravity = (0, self.gravity_level)
        self.space.on_collision(ObjectCategory.PLAYER, ObjectCategory.ROCK, begin = self._rock_collision)
        self.space.on_collision(ObjectCategory.PLAYER, ObjectCategory.TARGET, begin = self._target_collision)

        self.curr_step = 0
        self.hit_rock = False
        self.hit_target = False

        self.space.gravity = (0, self.gravity_level)
        self.space.damping = 0.7

        return self._get_obs(), {}

    def _generate_player_pos(self) -> tuple[int, int]:

        rand_with_min = 50
        rand_with_max = max(self.metadata['map_settings']['width'] - 50, 100)

        player_height_pos = max(self.metadata['map_settings']['height'] - 50, 100)

        return int(self.np_random.integers(low=rand_with_min, high=rand_with_max)), player_height_pos

    def _generate_target_pos(self) -> tuple[int, int]:

        rand_with_min = 50
        rand_with_max = max(self.metadata['map_settings']['width'] - 50, 100)

        target_height_pos = 50

        return int(self.np_random.integers(low=rand_with_min, high=rand_with_max)), target_height_pos

    def _generate_rock_pos(self) -> tuple[int, int]:

        rand_with_min = 0
        rand_with_max = max(self.metadata['map_settings']['width'], 100)

        rock_height_pos = 50

        return int(self.np_random.integers(low=rand_with_min, high=rand_with_max)), rock_height_pos


    def _create_target(self) -> tuple[pymunk.Circle, pymunk.Body]:
        target_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        target_body.position = self._generate_target_pos()
        target = pymunk.Circle(target_body, radius=self.target_size)
        target.filter = pymunk.ShapeFilter(categories=ObjectCategory.TARGET)
        target.sensor = True
        target.color = (0, 0, 0, 0)
        target.collision_type = ObjectCategory.TARGET

        return target, target_body


    def _create_player(self) -> tuple[pymunk.Circle, pymunk.Body]:
        player_body = pymunk.Body(mass=1,
                                  moment=pymunk.moment_for_circle(1, 0, self.player_size),
                                  body_type=pymunk.Body.DYNAMIC
                                  )

        player_body.position = self._generate_player_pos()

        player = pymunk.Circle(player_body, radius=self.player_size)
        player.color = (0, 0, 0, 0)

        player.filter = pymunk.ShapeFilter(categories=ObjectCategory.PLAYER)
        player.collision_type = ObjectCategory.PLAYER

        return player, player_body


    def _create_walls(self) -> list[pymunk.Segment]:

        map_height = self.metadata['map_settings']['height']
        map_width = self.metadata['map_settings']['width']

        floor = pymunk.Segment(self.space.static_body, (0, map_height), (map_width, map_height), 10)
        ceiling = pymunk.Segment(self.space.static_body, (0, 0), (map_width, 0), 10)
        left_wall = pymunk.Segment(self.space.static_body, (0, 0), (0, map_height), 10)
        right_wall = pymunk.Segment(self.space.static_body, (map_width, 0), (map_width, map_height), 10)

        walls = [floor, ceiling, left_wall, right_wall]

        for wall in walls:
            wall.friction = 1.0
            wall.elasticity = 0.2
            wall.filter = pymunk.ShapeFilter(categories=ObjectCategory.WALL)

        return walls

    def _create_pickaxe(self) -> pymunk.PinJoint:
        pos = self.player.body.position
        anchor_point = (pos.x, pos.y - self.pickaxe_length)

        joint = pymunk.PinJoint(
            self.space.static_body,
            self.player.body,
            anchor_point,
            (0, 0)
        )
        return joint


    def _create_rock(self) -> tuple[pymunk.Circle, pymunk.Body]:
        rock_body = pymunk.Body(mass=self.rock_mass,
                                moment=pymunk.moment_for_circle(self.rock_mass, 0, self.rock_size),
                                body_type=pymunk.Body.DYNAMIC
                                )

        rock_body.position = self._generate_rock_pos()

        rock = pymunk.Circle(rock_body, radius=self.rock_size)
        rock.color = (0, 0, 0, 0)
        rock.filter = pymunk.ShapeFilter(categories=ObjectCategory.ROCK)
        rock.collision_type = ObjectCategory.ROCK

        return rock, rock_body


    def _get_obs(self):
        obs = np.zeros(22, dtype=np.float32)

        obs[0] = self.player.body.position.x / 800.0
        obs[1] = self.player.body.position.y / 1000.0

        obs[2] = self.player.body.velocity.x / 500.0
        obs[3] = self.player.body.velocity.y / 500.0

        dx, dy = _get_relative_pos(self.player.body, self.target.body)
        obs[4] = dx / 800.0
        obs[5] = dy / 1000.0

        shape_filter = pymunk.ShapeFilter(categories=ObjectCategory.ROCK)
        rays = 12
        angle = 360 / rays

        for i in range(rays):
            rad_deg = np.deg2rad(angle * i)

            dx = np.cos(rad_deg) * self.ray_distance
            dy = np.sin(rad_deg) * self.ray_distance

            start_p = self.player.body.position
            end_p = (start_p.x + dx, start_p.y + dy)

            query = self.space.segment_query_first(start_p, end_p, 1, shape_filter)

            if query is None:
                obs[6 + i] = 1.0
            else:
                obs[6 + i] = query.alpha

        obs[18] = 1.0 if self.joint is not None else 0.0

        angle = self.player.body.angle
        obs[19] = np.sin(angle)
        obs[20] = np.cos(angle)

        obs[21] = self.player.body.angular_velocity / 10.0

        return obs

    def _rock_collision(self, arbiter, space, data):
        self.hit_rock = True
        print("GAME LOST - hit_rock")

    def _target_collision(self, arbiter, space, data):
        self.hit_target = True
        print("GAME WON - hit_target")
