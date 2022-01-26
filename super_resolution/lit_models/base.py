import argparse
import pytorch_lightning as pl 
import torch
from torchmetrics import Metric
import torch.nn.functional as F 

OPTIMIZER = 'SGD'
LR = 1e-4
MOMENTUM = 0.9
LOSS = 'mse_loss'
EPOCHS = 1000
ONE_CYCLE_TOTAL_STEPS = 100

def psnr(preds,target):
  mse = F.mse_loss(preds,target)
  psnr = 10*torch.log10(1/mse)
  return psnr

class PSNR(Metric):
    '''
    Custom PSNR metric implmenting the Pytorch Lightning's metric interface
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

class BaseLitModule(pl.LightningModule):
    '''
    Generic Pytorch-Lightning class that must by initialized with a pytorch module
    '''
    def __init__(self,model,args : argspace.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer",OPTIMIZER)
        self.optimizer_class = getattr(torch.optim,optimizer)

        self.lr = self.args.get("lr",LR)
        self.momentum = self.args.get("momentum",MOMENTUM)

        loss = self.args.get("loss",LOSS)
        self.loss_fn = getattr(torch.nn.functional,loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_psnr = PSNR()
        self.val_psnr = PSNR()
        self.test_psnr = PSNR()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer",type = str,default = OPTIMIZER,help = "optimizer class from torch.optim")
        parser.add_argument("--lr",type = float,default = LR)
        parser.add_argument("--one_cycle_max_lr",type = float,default = None)
        parser.add_argument("--one_cycle_total_steps",type = int,default = ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss",type = str,default = LOSS,help = "loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer, max_lr = self.one_cycle_max_lr, total_steps = self.one_cycle_total_steps
        )
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor': 'val_loss'}

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        xb,yb = batch
        preds = self(xb)
        loss = self.loss_fn(preds,yb)
        self.log("train_loss",loss)
        self.train_psnr(preds,yb)
        self.log("train_psnr",self.train_psnr,on_step = False, on_epoch = True)
        return loss

    def validation_step(self,batch,batch_idx):
        xb,yb = batch
        preds = self(xb)
        loss = self.loss_fn(preds,yb)
        self.log("val_loss",loss,prog_bar = True)
        self.val_psnr(preds,yb)
        self.log("val_psnr",self.val_psnr,on_step = False, on_epoch = True,prog_bar = True)
      
    def test_step(self,batch,batch_idx):
        xb,yb = batch
        preds = self(xb)
        self.test_psnr(preds,yb)
        self.log("test_psnr",self.test_psnr,on_step = False, on_epoch = True)
