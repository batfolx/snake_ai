import pygame


from constants import *
from game_utils import Rectangle, Snake
import random
from agent import SnakeAgent, dense_layer
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
import time
from tf_agents.utils import common
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import utils
from snake_env import SnakeGameEnv


def game(display, agent: SnakeAgent):
    """
    Automated snake game using learning
    :param agent: The agent used for reinforcement learning
    :param display:
    :return:
    """

    apple_x, apple_y = choose_new_apple()
    s = Snake()
    s.segments.insert(0, Rectangle(display, black, [s.x, s.y, RECT_SIZE, RECT_SIZE]))
    score = 0
    font = pygame.font.SysFont("Times New Roman", 24)

    # take one step to get the initial observation of the game
    step = game_step(display, s, apple_x, apple_y, font, score)

    # get the timestep and action spec for this particular environment of the game
    timestep, action_spec = agent.get_time_step(s, apple_x, apple_y, True)
    fc_layer_params = (100, 50)


    action_tensor_spec = tensor_spec.from_spec(action_spec)
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    train_step_counter = tf.Variable(0)

    agent.agent = dqn_agent.DqnAgent(
        time_step_spec=timestep,
        action_spec=action_spec,
        optimizer=agent.optimizer,
        train_step_counter=train_step_counter,
        q_network=q_net,
        td_errors_loss_fn=common.element_wise_squared_loss,
    )
    agent.agent.initialize()
    # eval_policy = agent.agent.policy
    # collect_policy = agent.agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(timestep,
                                                    action_spec)
    random_policy.action(timestep)

    while step:
        step = game_step(display, s, apple_x, apple_y, font, score)
        print(agent.get_observation(s, apple_x, apple_y))
        time.sleep(0.5)

    return score


def game_step(display,
              s: Snake,
              apple_x,
              apple_y,
              font,
              score):
    """
    Takes one step in the snake game
    :param display: The games display
    :param s: The snake player
    :param apple_x: The position of the apple x coordinate
    :param apple_y: The position of the apple y coordinate
    :param font:
    :param score: The score
    :return:
    """
    move = random.randint(0, 3)
    if move == UP:
        x_step = 0
        y_step = -STEP
        s.change_direction(UP)
    elif move == DOWN:
        x_step = 0
        y_step = STEP
        s.change_direction(DOWN)
    elif move == LEFT:
        x_step = -STEP
        y_step = 0
        s.change_direction(LEFT)
    else: #move == RIGHT:
        x_step = STEP
        y_step = 0
        s.change_direction(RIGHT)

    # calculate the training inputs for the next move

    s.x += x_step
    s.y += y_step

    if s.is_collision():
        return False

    if s.did_get_food(apple_x, apple_y):
        score += SCORE_STEP
        apple_x, apple_y = choose_new_apple()
        s.insert_segment(Rectangle(display, black, [s.x - x_step, s.y - y_step, RECT_SIZE, RECT_SIZE]))

    display.fill(white)
    s.pop_segment()
    s.insert_segment(Rectangle(display, black, [s.x, s.y, RECT_SIZE, RECT_SIZE]))
    for seg in s.segments:
        pygame.draw.rect(seg.display, black, [seg.rect[0], seg.rect[1], RECT_SIZE, RECT_SIZE])

    pygame.draw.rect(display, red, [apple_x, apple_y, RECT_SIZE, RECT_SIZE])
    display.blit(font.render(f'Score {score}', True, black), (0, 0))
    pygame.display.update()

    return True


def choose_new_apple() -> (int, int):
    """
    Gets a tuple of numbers between height and width to the nearest multiple 5 of
    :return:
    """
    return BASE * round(random.randint(0, (WIDTH - RECT_SIZE)) / BASE), \
           BASE * round(random.randint(0, HEIGHT - RECT_SIZE)) / BASE


def setup():
    pygame.init()
    display = pygame.display.set_mode((HEIGHT, WIDTH))
    pygame.display.set_caption("Snake")
    font = pygame.font.SysFont("Times New Roman", 24)
    env = SnakeGameEnv(display, font)
    utils.validate_py_environment(env, 1)




def setup_agent() -> SnakeAgent:
    a = SnakeAgent()


    return a


setup()
pygame.quit()
