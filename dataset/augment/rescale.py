import numbers
import random
import cv2
import numpy as np

def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)


def get_interpolation(type='random'):
    if type == 'random':
        choice = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        interpolation = choice[random.randint(0, len(choice) - 1)]
    elif type == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif type == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif type == 'cubic':
        interpolation = cv2.INTER_CUBIC
    elif type == 'area':
        interpolation = cv2.INTER_AREA
    else:
        raise TypeError('Interpolation types only nearest, linear, cubic, area are supported!')
    return interpolation

class Rescale(object):

    def __init__(self, factor=4, base_size=(128, 512)):
        """ Define image scales using gaussian pyramid and rescale image to target scale.
        
        Args:
            factor: the decayed factor from base size, factor=4 keeps target scale by default.
            base_size: base size the build the bottom layer of pyramid
        """
        if isinstance(factor, numbers.Number):
            self.factor = round(sample_uniform(0, factor))
        elif isinstance(factor, (tuple, list)) and len(factor) == 2:
            self.factor = round(sample_uniform(factor[0], factor[1]))
        else:
            raise Exception('factor must be number or list with length 2')
        # assert factor is valid
        self.base_h, self.base_w = base_size[:2]

    def __call__(self, img):
        if self.factor == 0: return img
        src_h, src_w = img.shape[:2]
        cur_w, cur_h = self.base_w, self.base_h
        scale_img = cv2.resize(img, (cur_w, cur_h), interpolation=get_interpolation())
        for _ in range(self.factor):
            scale_img = cv2.pyrDown(scale_img)
        scale_img = cv2.resize(scale_img, (src_w, src_h), interpolation=get_interpolation())
        return scale_img
