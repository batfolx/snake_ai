import numpy as np
from game_utils import Snake
from constants import *
from parameters import *
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.specs import ArraySpec, BoundedArraySpec, TensorSpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import time_step_spec


class SnakeAgent:
    def __init__(self):

        self.reward = 0
        self.gamma = 0.9
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.1
        self.time_step_spec = None

        # Adam optimizer, the opposite of an Eve optimizer.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.agent = None

    def set_dqn_agent(self, a):
        self.agent = a

    def get_time_step(self):
        obs_spec = TensorSpec(shape=(12,),
                              dtype=np.int32,
                              name="observation",

                              )

        # scalar float value
        reward_spec = TensorSpec((1,), np.dtype('float32'), 'reward')

        # scalar integer with 4 possible values (up, down, left, right)
        action_spec = BoundedTensorSpec((1,), np.dtype('int32'), minimum=0, maximum=3, name='action')
        #timestep = TimeStep(step_type=step_type, observation=obs_spec, reward=reward_spec, discount=0)
        timestep = time_step_spec(obs_spec)
        self.time_step_spec = timestep
        print('Observation spec:')
        print(obs_spec)
        print('Reward spec:')
        print(reward_spec)
        print('Action Spec:')
        print(action_spec)
        print("Time step:")
        print(timestep)
        return timestep, action_spec

    def get_observation(self, snake: Snake, food_x, food_y):
        """
        Gets the observation of the snake (the inputs to the neural network)
        :param food_y: the position of the food at y at the current frame
        :param food_x: the position of the food at x at the current frame
        :param snake: the snake object used in the game
        :return: a list of observations, either 1 or 0
        """
        # 0th -> snake left direction
        # 1th -> snake right direction
        # 2nd -> snake up direction
        # 3rd -> snake down direction
        state = []

        # 1 for true, 0 for false
        if snake.direction() == LEFT:
            state.append(1)
        else:
            state.append(0)

        if snake.direction() == RIGHT:
            state.append(1)
        else:
            state.append(0)

        if snake.direction() == UP:
            state.append(1)
        else:
            state.append(0)

        if snake.direction() == DOWN:
            state.append(1)
        else:
            state.append(0)

        # 4th -> food is left
        # 5th -> food is right
        # 6th -> food is above
        # 7th -> food is below
        if snake.x < food_x:
            state.append(1)
        else:
            state.append(0)

        if snake.x > food_x:
            state.append(1)
        else:
            state.append(0)

        # y is inversed, therefore if snake y is less than food_y, it is above
        if snake.y < food_y:
            state.append(1)
        else:
            state.append(0)

        # y is below
        if snake.y >= food_y:
            state.append(1)
        else:
            state.append(0)

        # danger -> left 1 block
        # danger -> right 1 block
        # danger -> up 1 block
        # danger -> down 1 block
        if snake.x - STEP < 0:
            state.append(1)
        else:
            state.append(0)

        if snake.x + STEP > WIDTH:
            state.append(1)
        else:
            state.append(0)

        # 0 is at the top, HEIGHT is at the bottom, so if snake y is 50 - STEP < 0, then wall is above
        if snake.y - STEP < 0:
            state.append(1)
        else:
            state.append(0)

        # danger of hitting a wall
        if snake.y + STEP > snake.max_y:
            state.append(1)
        else:
            state.append(0)
        return state


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))
