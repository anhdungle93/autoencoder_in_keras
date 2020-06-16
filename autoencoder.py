import numpy as np
from tensorflow import set_random_seed

def seedy(s):
    np.random.seed(s)
    set_random_seed(s)