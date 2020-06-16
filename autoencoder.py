import numpy as np
from tensorflow import set_random_seed

#  set the random seed for numpy and tensorflow backend
#  to have a more consistent testing environment
def seedy(s):
    np.random.seed(s)
    set_random_seed(s)

# encoding dimension is the length of input data
class AutoEncoder:
    def __init__(self, encoding_dim=3):
        self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)
        self.x = np.array([[r(), r(), r()] for _ in range(1000)])
        print(self.x)
    

