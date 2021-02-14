### game constants ###
HEIGHT = 900
WIDTH = 900
FPS = 15
SCORE_STEP = 10

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

COLLIDE_PENALTY = -100
FOOD_REWARD = 25
STEP_PENALTY = -1.5
DISCOUNT = 1.0

BASE = 50
RECT_SIZE = BASE
STEP = BASE


### Agent constants ###

NUM_ITERATIONS = 75000  # @param {type:"integer"}
INITIAL_COLLECT_STEPS = 100  # @param {type:"integer"}
COLLECT_STEPS_PER_ITERATION = 1  # @param {type:"integer"}
REPLAY_BUFFER_MAX_LEN = 100000  # @param {type:"integer"}

BATCH_SIZE = 64  # @param {type:"integer"}
LEARNING_RATE = 1e-3  # @param {type:"number"}
LOG_INTERVAL = 200  # @param {type:"integer"}

NUM_EVAL_EPISODES = 10  # @param {type:"integer"}
EVAL_INTERVAL = 1000  # @param {type:"integer"}