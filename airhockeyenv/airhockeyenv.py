#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Box2D import b2World, b2PolygonShape, b2CircleShape, b2FixtureDef
from os import environ
from time import time, sleep
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import numpy as np
import gymnasium as gym


BG_COLOR = (0,0,0)


class AirHockey2D(gym.Env):
    
    metadata = {
        "render_modes": ["human", "interactive", "rgb_array"],
        "render_fps": 60,
        "window_size": (640, 480),
        "scale_factor": 150
    }
    
    LINK1_LENGTH = 1
    LINK2_LENGTH = 0.5
    LINK_WIDTH = 0.02
    MALLET_RADIUS = .04815
    
    MAX_THETA = np.pi
    SAFETY_MARGIN = 0.05
    MAX_ANGULAR_SPEED = 5*np.pi
    MAX_TORQUE = 10
    MAX_COORD = LINK1_LENGTH + LINK2_LENGTH
    MAX_TARGET_SPEED = 10
    
    SIMULATION_HZ = 1000
    SIMULATION_TIMESTEP = 1/SIMULATION_HZ
    
    def __init__(self, epsilon=1e-1, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        self.render_mode = render_mode
        
        self.observation_space = gym.spaces.Dict(
            {
                "joints": gym.spaces.Box(
                    low=np.array([-self.MAX_THETA, -self.MAX_THETA, -self.MAX_ANGULAR_SPEED, -self.MAX_ANGULAR_SPEED, -np.inf, -np.inf]),
                    high=np.array([self.MAX_THETA, self.MAX_THETA, self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED, np.inf, np.inf]),
                    dtype=np.float64
                ),
                "cartesian": gym.spaces.Box(
                    low=np.array([ -self.MAX_COORD, -self.MAX_COORD, -np.inf, -np.inf, -np.inf, -np.inf]),
                    high=np.array([self.MAX_COORD, self.MAX_COORD, np.inf, np.inf, np.inf, np.inf]),
                    dtype=np.float64
                ),
                "target":gym.spaces.Box(
                    low=np.array([ -self.MAX_COORD, -self.MAX_COORD, -self.MAX_TARGET_SPEED, -self.MAX_TARGET_SPEED]),
                    high=np.array([self.MAX_COORD, self.MAX_COORD, self.MAX_TARGET_SPEED, self.MAX_TARGET_SPEED]),
                    dtype=np.float64
                )
            }
        )
        
        self.action_space = gym.spaces.Box(low=np.array([-1.0,-1.0]), high=np.array([1.0, 1.0]), dtype=np.float64)
        
        self._user_exit = False
        
        self._prev_qd = np.zeros(2)
        self._prev_xd = np.zeros(2)
        self._target = np.zeros(4)
        self._epsilon = epsilon
        
        self._world = b2World(gravity=(0,0))
        self._origin = self._world.CreateStaticBody(
            position=(0,0)
        )
        
        link_density = 10.0 / self.LINK1_LENGTH
        self._link1 = self._world.CreateDynamicBody(
            position=(self.LINK1_LENGTH/2,0),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(self.LINK1_LENGTH/2, self.LINK_WIDTH/2)),
                density=link_density
            )
        )
        self._link2 = self._world.CreateDynamicBody(
            position=(self.LINK1_LENGTH+self.LINK2_LENGTH/2,0),
            fixtures=[
                b2FixtureDef(
                    shape=b2PolygonShape(box=(self.LINK2_LENGTH/2, self.LINK_WIDTH/2)),
                    density=link_density
                ),
                b2FixtureDef(
                    shape=b2CircleShape(pos=(self.LINK2_LENGTH/2, 0),radius=self.MALLET_RADIUS),
                    density=.1/(np.pi*self.MALLET_RADIUS**2),
                    restitution=0
                )
            ]
        )
        self._joint1 = self._world.CreateRevoluteJoint(
            bodyA=self._origin,
            bodyB=self._link1,
            anchor=self._origin.worldCenter,
            lowerAngle=-(self.MAX_THETA-self.SAFETY_MARGIN),
            upperAngle=self.MAX_THETA-self.SAFETY_MARGIN,
            enableLimit=True,
            maxMotorTorque=self.MAX_TORQUE,
            enableMotor=True
        )
        self._joint2 = self._world.CreateRevoluteJoint(
            bodyA=self._link1,
            bodyB=self._link2,
            anchor=(self.LINK1_LENGTH, 0),
            lowerAngle=-(self.MAX_THETA-self.SAFETY_MARGIN),
            upperAngle=self.MAX_THETA-self.SAFETY_MARGIN,
            enableLimit=True,
            maxMotorTorque=self.MAX_TORQUE,
            enableMotor=True
        )
        self._window = None
        self._canvas = None
        self._time_point_last_frame = 0
        self._time_point_last_update = 0
        self._interactive_controls = np.zeros(2)
    
    def distance_to_target(self):
        return np.linalg.norm(self.x - self._target[:2])
        
    def difference_with_target_speed(self):
        return np.linalg.norm(self.xd - self._target[2:4])
    
    @property
    def done(self):
        return self.distance_to_target() < self._epsilon and self.difference_with_target_speed() < self._epsilon
    
    @property
    def q(self):
        return np.array([self._joint1.angle, self._joint2.angle])
        
    @property
    def qd(self):
        return np.array([self._joint1.speed, self._joint2.speed])
        
    @property
    def qdd(self):
        return self.SIMULATION_HZ*(self.qd - self._prev_qd)
    
    @property
    def x(self):
        return self._fk()

    @property
    def xd(self):
        return self._jacobian()

    @property
    def xdd(self):
        return self.SIMULATION_HZ*(self.xd - self._prev_xd)
    
    def _get_obs(self):
        return {
            "joints": np.hstack((self.q, self.qd, self.qdd)),
            "cartesian": np.hstack((self.x, self.xd, self.xdd)),
            "target": self._target
        }
        
    def _get_info(self):
        return {
            "distance_to_target": self.distance_to_target(),
            "difference_with_target_speed": self.difference_with_target_speed()
        }
    
    def _jacobian(self, q=None, qd=None):
        if q is None: q = self.q
        if qd is None: qd = self.qd
        dx = - self.LINK1_LENGTH*qd[0]*np.sin(q[0]) - self.LINK2_LENGTH*np.sum(qd)*np.sin(np.sum(q))
        dy = self.LINK1_LENGTH*qd[0]*np.cos(q[0]) + self.LINK2_LENGTH*np.sum(qd)*np.cos(np.sum(q))
        return np.array((dx,dy))
    
    def _fk(self, q=None):
        if q is None: q = self.q
        x = self.LINK1_LENGTH*np.cos(q[0]) + self.LINK2_LENGTH*np.cos(np.sum(q))
        y = self.LINK1_LENGTH*np.sin(q[0]) + self.LINK2_LENGTH*np.sin(np.sum(q))
        return np.array((x,y))
        
    def _sample_angles(self):
        return self.np_random.uniform(-self.MAX_THETA, self.MAX_THETA, 2)
        
    def _is_reachable_position(self, position):
        # TODO
        return True

    def _random_reachable_position(self):
        reachable = False
        position = None
        while not reachable:
            q = self._sample_angles()
            position = self._fk(q)
            reachable = self._is_reachable_position(position)
        return position
        
    def _screen_coordinates(self, point):
        scale_factor = self.metadata["scale_factor"]
        w, h = self.metadata["window_size"]
        x, y = point
        return (w/2 + scale_factor*x, h/2 - scale_factor*y)
        
    def _draw_polygon(self, canvas, color, transform, points):
        transformed_points = [self._screen_coordinates(transform*point) for point in points]
        pygame.draw.polygon(canvas, color, transformed_points)
        
    def _draw_circle(self, canvas, color, center, radius):
        scale_factor = self.metadata["scale_factor"]
        center = self._screen_coordinates(center)
        pygame.draw.circle(canvas, color, center, scale_factor*radius)
        
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or ( event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE ):
                self._user_exit = True
            elif event.type == pygame.KEYDOWN and self.render_mode == "interactive":
                if event.key == pygame.K_w: self._interactive_controls[0] = 1
                elif event.key == pygame.K_s: self._interactive_controls[0] = -1
                elif event.key == pygame.K_d: self._interactive_controls[1] = 1
                elif event.key == pygame.K_a: self._interactive_controls[1] = -1
            elif event.type == pygame.KEYUP and self.render_mode == "interactive":
                if event.key == pygame.K_w: self._interactive_controls[0] = 0
                elif event.key == pygame.K_s: self._interactive_controls[0] = 0
                elif event.key == pygame.K_d: self._interactive_controls[1] = 0
                elif event.key == pygame.K_a: self._interactive_controls[1] = 0
                elif event.key == pygame.K_r: self.reset()
    
    def _frame_pending(self):
        fps = self.metadata["render_fps"]
        time_since_last_frame = time() - self._time_point_last_frame
        return time_since_last_frame >= 1. / fps
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        window_size = self.metadata["window_size"]
        self._time_point_last_frame = time()
        if self.render_mode in ("human", "interactive"):
            if self._window is None:
                pygame.init()
                self._window = pygame.display.set_mode(window_size)
            self._handle_events()
        canvas = pygame.Surface(window_size)
        canvas.fill(BG_COLOR)
        objects = (self._link1, self._link2)
        joints = (self._joint1, self._joint2)
        for obj in objects:
            for fixture in obj.fixtures:
                if isinstance(fixture.shape, b2PolygonShape):
                    self._draw_polygon(canvas, "gray", obj.transform, fixture.shape.vertices)
                elif isinstance(fixture.shape, b2CircleShape):
                    self._draw_circle(canvas, "gray", obj.GetWorldPoint(fixture.shape.pos), fixture.shape.radius)
        target = self._target[:2]
        target_color = "green" if self.done else "blue"
        self._draw_circle(canvas, target_color, target, 0.05)
        for joint in joints:
            self._draw_circle(canvas, "red", joint.anchorA, 0.05)
        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))
        else:
            self._window.blit(canvas, canvas.get_rect())
            pygame.display.update()
        
    def close(self):
        if self._window is not None:
            pygame.quit()
            self._window = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._world.ClearForces()
        for body in (self._link1, self._link2):
            body.linearVelocity = (0,0)
            body.angularVelocity = 0
            
        for joint in (self._joint1, self._joint2):
            joint.motorSpeed = 0
        
        q = self._sample_angles()
        
        self._prev_qd = np.zeros(2)
        self._prev_xd = np.zeros(2)
        
        self._link1.angle = q[0]
        self._link2.angle = q[0] + q[1]
        
        self._target[:2] = self._random_reachable_position()
        
        # perform some steps to let the simulation settle
        for _ in range(self.SIMULATION_HZ):
            self._world.Step(self.SIMULATION_TIMESTEP, 10, 10)
        
        if self.render_mode in ("human", "interactive"):
            self._render_frame()
            
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
        
    def step(self, action):
        if self.render_mode == "interactive":
            action = self._interactive_controls
            while time() - self._time_point_last_update < self.SIMULATION_TIMESTEP:
                pass
            self._time_point_last_update = time()
        
        self._joint1.motorSpeed = action[0]*self.MAX_ANGULAR_SPEED
        self._joint2.motorSpeed = action[1]*self.MAX_ANGULAR_SPEED
            
        self._world.Step(self.SIMULATION_TIMESTEP, 10, 10)
            
        obs = self._get_obs()
        info = self._get_info()
        
        self._prev_qd = obs["joints"][2:4]
        self._prev_xd = obs["cartesian"][2:4]
            
        done = self.done
        reward = -1 if done else 0
        
        if self.render_mode in ("human", "interactive") and self._frame_pending():
            self._render_frame()
        
        return (obs, reward, done, self._user_exit, info)
        
if __name__ == "__main__":
    from gymnasium.wrappers import RecordVideo
    with gym.make("airhockeyenv/MoveToTarget-v0", render_mode="rgb_array", max_episode_steps=2000) as env:
        env = RecordVideo(env, video_folder="videos", episode_trigger=lambda _: True, fps=1000)
        
        
        truncated = False
        env.reset()
        
        while not truncated:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
        env.close()
        
        