"""MNIST DataModule"""
import argparse

from torch.utils.data import random_split
from torchvision import transforms
from super_resolution.base_data_module import BaseDataModule

