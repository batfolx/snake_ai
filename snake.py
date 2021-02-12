import pygame

from constants import *
from parameters import *
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
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment

from tf_agents.environments import utils
from snake_env import SnakeGameEnv, CardGameEnv


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
    timestep, action_spec = agent.get_time_step()
    fc_layer_params = (100, 50)

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
    else:  # move == RIGHT:
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
    # snake_agent = SnakeAgent()
    # game(display, snake_agent)

    env = SnakeGameEnv(display, font)
    # env = CardGameEnv()
    # utils.validate_py_environment(env)
    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    env = tf_py_environment.TFPyEnvironment(env)

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # it's output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    agent.initialize()
    random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                    env.action_spec())
    time_step = env.reset()

    random_policy.action(time_step)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=REPLAY_BUFFER_MAX_LEN)

    collect_data(env, random_policy, replay_buffer, INITIAL_COLLECT_STEPS)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=BATCH_SIZE,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)
    print(iterator)
    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(env, agent.policy, NUM_EVAL_EPISODES)
    returns = [avg_return]

    for _ in range(NUM_ITERATIONS):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(env, agent.collect_policy, replay_buffer, COLLECT_STEPS_PER_ITERATION)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % LOG_INTERVAL == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(env, agent.policy, NUM_EVAL_EPISODES)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
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


setup()
pygame.quit()
