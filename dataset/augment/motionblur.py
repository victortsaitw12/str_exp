
import numbers
import numpy as np
import cv2

def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude

def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)

class MotionBlur(object):
    def __init__(self, degrees=12, angle=90):
        if isinstance(degrees, numbers.Number):
            self.degree = max(int(sample_asym(degrees)), 1)
        elif isinstance(degrees, (tuple, list)) and len(degrees) == 2:
            self.degree = int(sample_uniform(degrees[0], degrees[1]))
        else:
            raise Exception('degree must be number or list with length 2')
        self.angle = sample_uniform(-angle, angle)

    def __call__(self, img):
        M = cv2.getRotationMatrix2D((self.degree // 2, self.degree // 2), self.angle, 1)
        motion_blur_kernel = np.zeros((self.degree, self.degree))
        motion_blur_kernel[self.degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img