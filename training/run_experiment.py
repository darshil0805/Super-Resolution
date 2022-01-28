'''For running Experiemnts'''
import argparse
import importlib
import numpy as np 
import torch
import pytorch_lightning as pl 
import wandb

from super_resolution import lit_models

# Setting random seeds
np.random.seed(42)
torch.manual_seed(42)

def _import_class(module_and_class_name : str)-> type:
    '''Function to import class from a module ex: super_resolution.models.srcnn'''
    module_name,class_name = module_and_class_name.rsplit(".",1)
    module = importlib.import_module(module_name)
    class_ = getattr(module,class_name)
    return class_

def _setup_parser()