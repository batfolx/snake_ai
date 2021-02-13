import abc
from abc import ABC
import time
import tensorflow as tf
import random
import pygame
from game_utils import Snake, Rectangle
from agent import SnakeAgent
from constants import *
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import BoundedArraySpec


class SnakeGameEnv(py_environment.PyEnvironment, ABC):

    def __init__(self, display, font):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = BoundedArraySpec(
            shape=(12,), dtype=np.int32, minimum=0, maximum=1, name='observation')

        self.score = 0
        self.food_x, self.food_y = choose_new_apple()
        self.s = Snake()
        self.s.segments.insert(0, Rectangle(display, black, [self.s.x, self.s.y, RECT_SIZE, RECT_SIZE]))
        self.snake_agent = SnakeAgent()
        self.display = display
        self.font = font
        self.steps = 0
        self.game_number = 0
        self.did_collide = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.score = 0
        self.s.reset_snake()
        self.s.segments.insert(0, Rectangle(self.display, black, [self.s.x, self.s.y, RECT_SIZE, RECT_SIZE]))
        self.food_x, self.food_y = choose_new_apple()
        self.steps = 0
        self.game_number += 1
        self.did_collide = False
        obs = self.snake_agent.get_observation(self.s, self.food_x, self.food_y)

        return ts.restart(np.array(obs, dtype=np.int32))

    def _step(self, action):

        if self.steps > 50 or self.did_collide:
            return self.reset()

        move = action
        if move == UP:
            x_step = 0
            y_step = -STEP
            self.s.change_direction(UP)
        elif move == DOWN:
            x_step = 0
            y_step = STEP
            self.s.change_direction(DOWN)
        elif move == LEFT:
            x_step = -STEP
            y_step = 0
            self.s.change_direction(LEFT)
        elif move == RIGHT:
            x_step = STEP
            y_step = 0
            self.s.change_direction(RIGHT)
        else:
            raise ValueError("Action can only be 0-3 inclusive.")

        self.s.x += x_step
        self.s.y += y_step

        if self.s.is_collision():
            obs = self.snake_agent.get_observation(self.s, self.food_x, self.food_y)
            self.did_collide = True
            return ts.termination(np.array(obs, dtype=np.int32), -10)

        got_food = False

        if self.s.did_get_food(self.food_x, self.food_y):
            self.score += SCORE_STEP
            self.steps = 0
            got_food = True
            self.food_x, self.food_y = choose_new_apple()
            self.s.insert_segment(
                Rectangle(self.display, black, [self.s.x - x_step, self.s.y - y_step, RECT_SIZE, RECT_SIZE]))

        self.display.fill(white)
        self.s.pop_segment()
        self.s.insert_segment(Rectangle(self.display, black, [self.s.x, self.s.y, RECT_SIZE, RECT_SIZE]))
        for seg in self.s.segments:
            pygame.draw.rect(seg.display, black, [seg.rect[0], seg.rect[1], RECT_SIZE, RECT_SIZE])

        pygame.draw.rect(self.display, red, [self.food_x, self.food_y, RECT_SIZE, RECT_SIZE])
        self.display.blit(self.font.render(f'Score {self.score}, game number {self.game_number}', True, black), (0, 0))
        pygame.display.update()
        obs = self.snake_agent.get_observation(self.s, self.food_x, self.food_y)
        self.steps += 1
        return ts.transition(np.array(obs, dtype=np.int32), 100 if got_food else -0.1, 0)


def choose_new_apple() -> (int, int):
    """
    Gets a tuple of numbers between height and width to the nearest multiple 5 of
    :return:
    """
    return BASE * round(random.randint(0, (WIDTH - RECT_SIZE)) / BASE), \
           BASE * round(random.randint(0, HEIGHT - RECT_SIZE)) / BASE

class CardGameEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)