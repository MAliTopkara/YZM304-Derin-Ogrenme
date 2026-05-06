"""Augmentation and preprocessing pipelines (torchvision-based, RGB-safe)."""
from torchvision import transforms

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _to_rgb(img):
    """Some PNGs are RGBA — convert to RGB before tensor ops."""
    return img.convert("RGB") if img.mode != "RGB" else img


def build_train_transform():
    return transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform():
    return transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
