from abc import ABC
import random
import pygame
from game_utils import Snake, Rectangle
from agent import SnakeAgent
from constants import *
import numpy as np

from tf_agents.trajectories import trajectory
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
        self.s = Snake()
        self.s.segments.insert(0, Rectangle(display, black, [self.s.x, self.s.y, RECT_SIZE, RECT_SIZE]))

        self.food_x, self.food_y = choose_new_apple(self.s)
        self.snake_agent = SnakeAgent()
        self.display = display
        self.font = font
        self.steps = 0
        self.high_score = 0
        self.game_number = 0
        self.did_collide = False
        self.tot_steps = 250

    def action_spec(self):
        """
        Gets the action spec from this environment
        :return: The classes action spec
        """
        return self._action_spec

    def observation_spec(self):
        """
        Gets the observation spec with this environment
        :return: The observation spec
        """
        return self._observation_spec

    def _reset(self):
        """
        Resets the environment
        :return: The time spec, restarted
        """
        self.score = 0
        self.s.reset_snake()
        self.s.segments.insert(0, Rectangle(self.display, black, [self.s.x, self.s.y, RECT_SIZE, RECT_SIZE]))
        self.food_x, self.food_y = choose_new_apple(self.s)
        self.steps = 0
        self.game_number += 1
        self.did_collide = False
        obs = self.snake_agent.get_observation(self.s, self.food_x, self.food_y)

        return ts.restart(np.array(obs, dtype=np.int32))

    def _step(self, action):
        """
        Function to take a step in the Snake environment.
        :param action: The action to take [0,3] (up, down, left, right) -> respectively
        :return:
        """
        if self.steps > self.tot_steps or self.did_collide:
            # reset the environment if the amount of steps exceeds the allowed steps
            # or if the snake collided with the wall in the previous step
            return self.reset()

        # set the snakes direction here from the agent
        move = action
        curr_direction = self.s.direction()
        if move == UP:
            if curr_direction != DOWN:
                x_step = 0
                y_step = -STEP
                self.s.change_direction(UP)
            else:
                x_step = 0
                y_step = STEP
        elif move == DOWN:
            if curr_direction != UP:
                x_step = 0
                y_step = STEP
                self.s.change_direction(DOWN)
            else:
                x_step = 0
                y_step = -STEP
        elif move == LEFT:
            if curr_direction != RIGHT:
                x_step = -STEP
                y_step = 0
                self.s.change_direction(LEFT)
            else:
                x_step = STEP
                y_step = 0
        elif move == RIGHT:
            if curr_direction != LEFT:
                x_step = STEP
                y_step = 0
                self.s.change_direction(RIGHT)
            else:
                x_step = -STEP
                y_step = 0
        else:
            raise ValueError("Action can only be 0-3 inclusive.")

        # increment the x and y step to the snakes internal variables
        self.s.x += x_step
        self.s.y += y_step

        # check to see if the snake collided with the wall or with itself. if it did
        # then terminate this episode, and set the did_collide variable to true to
        # reset the environment next time step
        if self.s.is_collision():
            obs = self.snake_agent.get_observation(self.s, self.food_x, self.food_y)
            self.did_collide = True
            return ts.termination(np.array(obs, dtype=np.int32), COLLIDE_PENALTY)

        # check to see that the snake obtained some food
        got_food = self.s.did_get_food(self.food_x, self.food_y)

        # if the snake got food, update score, update high score (if surpassed)
        # reset steps to zero (snake is doing well, don't end episode)
        # and then insert another segment for the snake
        if got_food:
            self.score += SCORE_STEP
            if self.score > self.high_score:
                self.high_score = self.score
            self.steps = 0
            got_food = True
            self.food_x, self.food_y = choose_new_apple(self.s)
            self.s.insert_segment(
                Rectangle(self.display, black, [self.s.x - x_step, self.s.y - y_step, RECT_SIZE, RECT_SIZE]))

        # color the display white
        self.display.fill(white)

        # remove the head off the top
        self.s.pop_segment()

        # insert new position onto the head
        self.s.insert_segment(Rectangle(self.display, black, [self.s.x, self.s.y, RECT_SIZE, RECT_SIZE]))

        # redraw segments on the display
        for seg in self.s.segments:
            pygame.draw.rect(seg.display, black, [seg.rect[0], seg.rect[1], RECT_SIZE, RECT_SIZE])

        # draw the food on the screen
        pygame.draw.rect(self.display, red, [self.food_x, self.food_y, RECT_SIZE, RECT_SIZE])

        # display stats to screen
        self.display.blit(self.font.render(f'Score {self.score}, High Score {self.high_score},'
                                           f' Generation {self.game_number}, '
                                           f'Step {self.steps}/{self.tot_steps}', True, black), (0, 0))

        # update the UI
        pygame.display.update()

        # retrieve the observation for this current time step
        obs = self.snake_agent.get_observation(self.s, self.food_x, self.food_y)
        self.steps += 1
        #time.sleep(0.035)

        # return transition for environment with scaling reward, else a small step penalty if
        # the AI did not grab a food on this time step
        return ts.transition(np.array(obs, dtype=np.int32),
                             (FOOD_REWARD * self.score) if got_food else STEP_PENALTY, DISCOUNT)


def choose_new_apple(snake: Snake) -> (int, int):
    """
    Gets a tuple of numbers between height and width to the nearest multiple of {BASE}
    :return:
    """
    while True:
        x = BASE * round(random.randint(0, (WIDTH - RECT_SIZE)) / BASE)
        y = BASE * round(random.randint(0, (WIDTH - RECT_SIZE)) / BASE)
        is_good = True
        for segment in snake.segments:
            if segment.rect[0] - RECT_SIZE < x < segment.rect[0] + RECT_SIZE \
                    and segment.rect[1] - RECT_SIZE < y < segment.rect[1] + RECT_SIZE:
                is_good = False
                break
        if is_good:
            return x, y


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0

    for i in range(num_episodes):
        print(f'On episode number {i}')
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            #print(f'{time_step}')
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)
