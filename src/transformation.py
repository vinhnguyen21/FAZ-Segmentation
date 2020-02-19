#adding library
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import random

def train_transformation(image, mask, size):
    tensor_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=[-45, 45], translate=[0.15, 0.15], scale=[1.0, 1.2]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)], p=0.85),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=[-45, 45], translate=[0.15, 0.15], scale=[1.0, 1.2]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)], p=0.85),
        transforms.ToTensor()
    ])
    
    #apply transform to image and grouth truth:
    seed = np.random.randint(2147483647) # make a seed with numpy generator 
    random.seed(seed)
    image = tensor_transform(image)
    random.seed(seed) # apply this seed to target tranfsorms
    mask = mask_transform(mask)
#     mask = torch.ByteTensor(np.array(mask))

    return image, mask

def inference_transformation(image, mask, size):
    """ Transform image for validation

    Parameters
    ----------
    image: PIL.Image
        image to transform
    size: int
        size to scale

    Returns
    -------

    """
    tensor_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    #apply transform to image and grouth truth:
    image = tensor_transform(image)
    mask = mask_transform(mask)
#     mask = torch.ByteTensor(np.array(mask))
    return image, mask

def tensor_transform(image, size):
    tensor_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    return tensor_transforms(image)