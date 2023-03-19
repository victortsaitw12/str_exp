#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
from torchvision import transforms

class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5):
        self.p = p
        self.transforms = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                 saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transforms(img)
        else:
            return img