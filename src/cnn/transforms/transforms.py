import random
import math

import cv2
from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform


def resized_crop(image, height, width, x_min, y_min, x_max, y_max):
    image = F.crop(image, x_min, y_min, x_max, y_max)
    image = cv2.resize(image, (width, height))
    return image


class RandomResizedCrop(ImageOnlyTransform):

    def __init__(self, height, width, scale=(0.08, 1.0), ratio=(3/4, 4/3), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio

    def apply(self, image, **params):

        height, width = image.shape[:2]
        area = height * width
        
        for attempt in range(15):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(self.ratio) <= (h / w) <= max(self.ratio):
                w, h = h, w

            if w <= width and h <= height:
                x_min = random.randint(0, width - w)
                y_min = random.randint(0, height - h)
                return resized_crop(image, self.height, self.width, x_min, y_min, x_min+w, y_min+h)

        min_side = min(height, width)
        x_min = random.randint(0, width - min_side)
        y_min = random.randint(0, height - min_side)
        return resized_crop(image, self.height, self.width, x_min, y_min, x_min+min_side, y_min+min_side)


class RandomDicomNoise(ImageOnlyTransform):
    
    def __init__(self, limit=None, limit_ratio=None, always_apply=False, p=0.5):
        assert limit or limit_ratio
        super().__init__(always_apply, p)
        self.limit = limit
        self.limit_ratio = limit_ratio

    def apply(self, image, **params):
        if self.limit:
            value = random.uniform(-self.limit, self.limit)
            image += value
        else:
            ratio = random.uniform(1.0-self.limit_ratio, 1.0+self.limit_ratio)
            image *= ratio

        return image
