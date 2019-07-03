"""Hyperparameters for deep Q network"""

BATCH_SIZE = 32
RPL_MEM_SIZE = 1000
NET_UPDATE_FREQ = 10000
GAMMA = 0.9
EPS_START = 1
EPS_END = 0.1

"""Parameters for RMSProp """
LEARNING_RATE = 0.00025
GRAD_MOM = 0.95
SQ_GRAD_MOM = 0.95
MIN_SQ_GRAD = 0.01