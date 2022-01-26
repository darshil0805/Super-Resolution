import argparse
import pytorch_lightning as pl 
import torch
from torchmetrics import Metric
import torch.nn.functional as F 

OPTIMIZER = 'SGD'
LR = 1e-4
LOSS = 'mse_loss'
ONE_CYCLE_TOTAL_STEPS = 100

def psnr(preds,target):
  mse = F.mse_loss(preds,target)
  psnr = 10*torch.log10(1/mse)
  return psnr

class PSNR(Metric):
    '''
    Creating a custom PSNR metric implmenting the Pytorch Lightning's metric interface
    '''
    def __init__(self,dist_sync_on_step = False):
        super().__init__(dist_sync_on_step = dist_sync_on_step)

        self.add_state("batch_psnr",default = torch.tensor(0),dist_reduce_fx = "sum")
        self.add_state("total"),default = torch.tensor(0),dist_reduce_fx = "sum")

    def update(self,preds : torch.tensor,target : torch.tensor):
        preds,target = self._input_format(preds,target)
        assert preds.shape = target.shape

        self.batch_psnr += torch.sum(psnr(preds,target))
        self.total += target.numel()

    def compute(self):
        return self.correct.float()/self.total




