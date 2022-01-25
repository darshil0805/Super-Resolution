import argparse
import pytorch_lightning as pl 
import torch

OPTIMIZER = 'SGD'
LR = 1e-4
LOSS = 'mse_loss'
ONE_CYCLE_TOTAL_STEPS = 100

