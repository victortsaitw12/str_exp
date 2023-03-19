#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
from torchvision.transforms import transforms as T
import numpy as np
from PIL import Image
from .gaussiannoise import GaussianNoise
from .motionblur import MotionBlur
from .rescale import Rescale

class Deterioration(object):
    def __init__(self, var, degrees, factor, p=0.5):
        self.p = p
        transforms = []
        if var is not None:
            transforms.append(GaussianNoise(var=var))
        if degrees is not None:
            transforms.append(MotionBlur(degrees=degrees))
        if factor is not None:
            transforms.append(Rescale(factor=factor))

        random.shuffle(transforms)
        transforms = T.Compose(transforms)
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            return Image.fromarray(self.transforms(img))
        else:
            return img