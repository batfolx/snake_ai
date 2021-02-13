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
import numpy as np
from tf_agents.environments import utils
import matplotlib.pyplot as plt
from snake_env import SnakeGameEnv
import IPython
import base64
import imageio


def game(display):
    """
    Automated snake game using learning
    :param agent: The agent used for reinforcement learning
    :param display:
    :return:
    """
    s = Snake()
    s.segments.insert(0, Rectangle(display, black, [s.x, s.y, RECT_SIZE, RECT_SIZE]))
    apple_x, apple_y = choose_new_apple(s)
    score = 0
    font = pygame.font.SysFont("Times New Roman", 24)
    agent = SnakeAgent()
    step = True
    while step:
        step, apple_x, apple_y = game_step(display, s, apple_x, apple_y, font, score)
        obs = agent.get_observation(s, apple_x, apple_y)
        print(obs)
        time.sleep(0.1)

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
    curr_direction = s.direction()
    x_step = -1
    y_step = -1

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # up key pressed
            if event.key == pygame.K_UP:
                if curr_direction != DOWN:
                    x_step = 0
                    y_step = -STEP
                    s.change_direction(UP)
                else:
                    x_step = 0
                    y_step = STEP
            # down key pressed
            if event.key == pygame.K_DOWN:
                if curr_direction != UP:
                    x_step = 0
                    y_step = STEP
                    s.change_direction(DOWN)
                else:
                    x_step = 0
                    y_step = -STEP
            # left key pressed
            if event.key == pygame.K_LEFT:
                if curr_direction != RIGHT:
                    x_step = -STEP
                    y_step = 0
                    s.change_direction(LEFT)
                else:
                    x_step = STEP
                    y_step = 0
            # right key pressed
            if event.key == pygame.K_RIGHT:
                if curr_direction != LEFT:
                    x_step = STEP
                    y_step = 0
                    s.change_direction(RIGHT)
                else:
                    x_step = -STEP
                    y_step = 0

    if x_step == -1 and y_step == -1:
        if curr_direction == UP:
            x_step = 0
            y_step = -STEP
        elif curr_direction == DOWN:
            x_step = 0
            y_step = STEP
        elif curr_direction == RIGHT:
            x_step = STEP
            y_step = 0
        else:
            x_step = -STEP
            y_step = 0
    # calculate the training inputs for the next move

    s.x += x_step
    s.y += y_step

    if s.is_collision():
        return False, apple_x, apple_y

    if s.did_get_food(apple_x, apple_y):
        score += SCORE_STEP
        apple_x, apple_y = choose_new_apple(s)
        s.insert_segment(Rectangle(display, black, [s.x - x_step, s.y - y_step, RECT_SIZE, RECT_SIZE]))

    display.fill(white)
    s.pop_segment()
    s.insert_segment(Rectangle(display, black, [s.x, s.y, RECT_SIZE, RECT_SIZE]))
    for seg in s.segments:
        pygame.draw.rect(seg.display, black, [seg.rect[0], seg.rect[1], RECT_SIZE, RECT_SIZE])

    pygame.draw.rect(display, red, [apple_x, apple_y, RECT_SIZE, RECT_SIZE])
    display.blit(font.render(f'Score {score}', True, black), (0, 0))
    pygame.display.update()

    return True, apple_x, apple_y


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
            if segment.rect[0] - RECT_SIZE < x < segment.rect[0] + RECT_SIZE and segment.rect[1] - RECT_SIZE < y < segment.rect[1] + RECT_SIZE:
                is_good = False
                break
        if is_good:
            return x, y


def ai_game():
    pygame.init()
    display = pygame.display.set_mode((HEIGHT, WIDTH))
    pygame.display.set_caption("Snake")
    font = pygame.font.SysFont("Times New Roman", 24)
    # snake_agent = SnakeAgent()
    # game(display, snake_agent)

    train_env = SnakeGameEnv(display, font)
    eval_env = SnakeGameEnv(display, font)
    # env = CardGameEnv()
    # utils.validate_py_environment(env)
    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    train_env = tf_py_environment.TFPyEnvironment(train_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env)

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
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    agent.initialize()
    print('Initialized Agent')
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    print('Reset time spec')
    time_step = train_env.reset()
    random_policy.action(time_step)
    print('Successfully instantiated random policy')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=REPLAY_BUFFER_MAX_LEN)

    print('Created replay buffer, collecting data ... ')
    collect_data(train_env, random_policy, replay_buffer, INITIAL_COLLECT_STEPS)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=BATCH_SIZE,
        num_steps=2).prefetch(3)
    print('Collecting data complete')
    iterator = iter(dataset)
    # Reset the train step
    agent.train_step_counter.assign(0)
    avg_return = compute_avg_return(train_env, agent.policy, NUM_EVAL_EPISODES)
    returns = [avg_return]
    print('Beginning to train...')
    for i in range(NUM_ITERATIONS):
        collect_data(train_env, agent.collect_policy, replay_buffer, COLLECT_STEPS_PER_ITERATION)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()
        print(train_env.time_step_spec())

        #print(f"Training agent through iteration {(i / NUM_ITERATIONS) * 100:.2f}%...")
        if step % LOG_INTERVAL == 0:
            pass
            #print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(train_env, agent.policy, NUM_EVAL_EPISODES)
            #print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    pygame.quit()
    iterations = range(0, NUM_ITERATIONS + 1, EVAL_INTERVAL)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.savefig("test.png")
    plt.show()


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
            print(f'{time_step}')
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


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(policy, filename, env, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = env.reset()
            video.append_data(env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)
                video.append_data(env.render())
    return embed_mp4(filename)


def manual_game():
    pygame.init()
    display = pygame.display.set_mode((HEIGHT, WIDTH))
    pygame.display.set_caption("Snake")
    font = pygame.font.SysFont("Times New Roman", 24)
    game(display)

def main():
    #manual_game()
    ai_game()
    pygame.quit()

main()
