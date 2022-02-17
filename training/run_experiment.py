'''For running Experiemnts'''
import argparse
import importlib
import numpy as np 
import torch
import pytorch_lightning as pl 
import wandb

from super_resolution.lit_models import base

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
    parser.add_argument("--model_class", type=str, default="SRCNN")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"super_resolution.data.{temp_args.data_class}")
    model_class = _import_class(f"super_resolution.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    base.BaseLitModule.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def main():
    '''Function used to run the experiment'''

    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"super_resolution.data.{args.data_class}")
    model_class = _import_class(f"super_resolution.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)
    lit_model_class = base.BaseLitModule

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    # pylint: disable=no-member
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member



if __name__ == "__main__":
    main()
    
