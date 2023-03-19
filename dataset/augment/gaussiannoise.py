import numbers
import numpy as np

def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude

def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)

class GaussianNoise(object):
    def __init__(self, mean=0, var=20):
        self.mean = mean
        if isinstance(var, numbers.Number):
            self.var = max(int(sample_asym(var)), 1)
        elif isinstance(var, (tuple, list)) and len(var) == 2:
            self.var = int(sample_uniform(var[0], var[1]))
        else:
            raise Exception('degree must be number or list with length 2')

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.var ** 0.5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img
    