import numpy as np

from network import Network

class Policy:

    def __init__(self, state_size, action_size, hiden_layer_size=8, seed=42):
         
        np.random.seed(seed)

        self.w = [
            np.random.rand(state_size, hiden_layer_size),
            np.random.rand(hiden_layer_size, action_size)
        ]
        self.best_w = self.w 
        
    def add_noise(self, scale):

        self.w = [
            self.best_w[0] + np.random.normal(0, scale, self.best_w[0].size)\
                .reshape(*self.best_w[0].shape),
            self.best_w[1] + np.random.normal(0, scale, self.best_w[1].size)\
                .reshape(*self.best_w[1].shape)
        ]
    
    def act(self, state):

        output = np.matmul(
            np.matmul(state, self.w[0]),
            self.w[1])
        probas = softmax(output)

        return np.random.choice([0,1], p=probas)

def softmax(x):
    e_x = np.exp(np.array(x))
    return e_x / e_x.sum()