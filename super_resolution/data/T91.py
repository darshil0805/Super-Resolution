"""MNIST DataModule"""
import argparse

from torch.utils.data import random_split
from torchvision import transforms
from super_resolution.base_data_module import BaseDataModule

class T91(BaseDataModule):
    '''
    Creating a T91 DataModule inheriting from the BaseDataModule class
    '''
    def __init__(self,args : argparse.Namespace) -> None:
        super().__init__(args)
        