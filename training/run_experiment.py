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

def _setup_parser():
    '''
    Setting up argument parser with data, model, trainer and other arguments
    '''
    parser = argparse.ArgumentParser(add_help = False)

    # Addting trainer arguments
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--data_class", type=str, default="T91")
    parser.add_argument("--model_class", type=str, default="srcnn")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def main():
    '''Function used to run the experiment'''

    parser = _setup_parser()
    

