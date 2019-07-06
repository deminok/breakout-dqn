"""Hyperparameters for deep Q network"""

ACTION_SIZE = 4
NUM_EPISODES = 5000
BATCH_SIZE = 32
RPL_START_SIZE = 50000
RPL_MEM_SIZE = 100000
NET_UPDATE_FREQ = 500
UPDATE_FREQUENCY = 4
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_ANNEAL_STEPS = 100000

"""Parameters for RMSProp """
LEARNING_RATE = 0.00025
GRAD_MOM = 0.95
SQ_GRAD_MOM = 0.95
MIN_SQ_GRAD = 0.01

"""User Parameters"""
RENDER = False #watch the agent learn
RECORD = False #record sessions to ./videos/
EVAL = True #store evaluation graphs in ./eval/
EVAL_PERIOD = 10
LOGGING = False #log information about current training values to console
LOGGING_PERIOD = 10

