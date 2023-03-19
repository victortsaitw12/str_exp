#!/usr/bin/python
# -*- coding: UTF-8 -*-

from imgaug import augmenters as iaa

def get_augmentations():
    return iaa.Sequential([
            iaa.Invert(0.5),
            iaa.OneOf([
                iaa.ChannelShuffle(0.35),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.KMeansColorQuantization(),
                iaa.HistogramEqualization(),
                iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.MultiplyBrightness((0.5, 1.5)),
                iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                iaa.ChangeColorTemperature((1100, 10000))
            ]),
            iaa.OneOf([
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                iaa.OneOf([
                    iaa.GaussianBlur((0.5, 1.5)),
                    iaa.AverageBlur(k=(2, 6)),
                    iaa.MedianBlur(k=(3, 7)),
                    iaa.MotionBlur(k=5)
                ])
            ]),
            iaa.OneOf([
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.ImpulseNoise(0.1),
                iaa.MultiplyElementwise((0.5, 1.5))
            ])
        ])