import numpy as np
import random
from PIL import Image
from .randomrotation import RandomRotation
from .randomaffine import RandomAffine
from .randomperspective import RandomPerspective

class Geometry(object):
    def __init__(self, degrees=15, translate=(0.3, 0.3), scale=(0.5, 2.),
                 shear=(45, 15), distortion=0.5, p=0.5):
        self.p = p
        type_p = random.random()
        if type_p < 0.33:
            self.transforms = RandomRotation(degrees=degrees)
        elif type_p < 0.66:
            self.transforms = RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
        else:
            self.transforms = RandomPerspective(distortion=distortion)

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            return Image.fromarray(self.transforms(img))
        else:
            return img