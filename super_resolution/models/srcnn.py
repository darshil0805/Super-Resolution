from typing import Any,Dict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Setting default number of channels for the network
CH1 = 64
CH2 = 32

class SRCNN(nn.Module):
  def __init__(self,data_config: Dict[str,Any],args: argparse.Namespace = None) -> None:
    super().__init__()
    self.args = vars(args) if args is not None else {}
    ch1 = self.args.get('ch1',CH1)
    ch2 = self.args.get('ch2',CH2)
    self.network = nn.Sequential(
            
                nn.Conv2d(1,ch1,kernel_size = 3,padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch1,ch2,kernel_size = 3,padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch2,1,kernel_size = 3,padding = 1),
                nn.ReLU(inplace=True)
            
        )

  def forward(self,xb):
    return self.network(xb)

  @staticmethod
  def add_to_argparse(parser):
    parser.add_argument("--ch1",type = int,default = 64)
    parser.add_argument("--ch2",type = int,default = 32)
    return parser
