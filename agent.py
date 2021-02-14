from game_utils import Snake
from constants import *
import tensorflow as tf


class SnakeAgent:
    def __init__(self):
        self.time_step_spec = None

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

        # danger -> left 1 block, whether it be itself or a block
        # danger -> right 1 block
        # danger -> up 1 block
        # danger -> down 1 block
        left_d, right_d, up_d, down_d = self.check_potential_collision_with_self(snake)
        if snake.x <= 0 or left_d:
            state.append(1)
        else:
            state.append(0)

        if snake.x + STEP >= snake.max_x or right_d:
            state.append(1)
        else:
            state.append(0)

        # 0 is at the top, HEIGHT is at the bottom, so if snake y is 50 - STEP < 0, then wall is above
        if snake.y - STEP <= 0 or up_d:
            state.append(1)
        else:
            state.append(0)

        # danger of hitting a wall
        if snake.y + STEP >= snake.max_y or down_d:
            state.append(1)
        else:
            state.append(0)

        return state

    def check_potential_collision_with_self(self, snake: Snake) -> (bool, bool, bool, bool):
        """
        Checks to see if the snake will have a collision
        with itself within the next time step
        """

        direction = snake.direction()
        left_d, right_d, up_d, down_d = False, False, False, False
        for i, segment in enumerate(snake.segments):
            # forget the first segment, is the head
            if i == 0:
                continue

            # # check to see if the snake will collide with left of itself
            if snake.x - STEP == segment.rect[0] and segment.rect[1] == snake.y:
                left_d = True

            # check to see if the snake will collide with right of itself
            if snake.x + STEP == segment.rect[0] and segment.rect[1] == snake.y:
                right_d = True

            # check to see if the snake will collide with itself above itself
            if snake.y - STEP == segment.rect[1] and segment.rect[0] == snake.x:
                up_d = True

            # check to see if the snake will collide with itself below itself
            if snake.y + STEP == segment.rect[1] and segment.rect[0] == snake.x:
                down_d = True

        # this lets the snake know that it has no danger
        # behind itself
        if direction == LEFT:
            right_d = False
        elif direction == RIGHT:
            left_d = False
        elif direction == UP:
            down_d = False
        elif direction == DOWN:
            up_d = False

        return left_d, right_d, up_d, down_d


def dense_layer(num_units):
    """
    Meant for adding layers for the neural network
    """
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))
