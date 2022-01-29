"""MNIST DataModule"""
import argparse
import zipfile
from torch.utils.data import random_split
from torchvision import transforms
from super_resolution.base_data_module import BaseDataModule
from torch.utils.data import Dataset
from PIL import Image

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order


opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

path_train = '../T91/train.zip'
path_test = '../T91/valid.zip'

def extract_T91(path):
    with zipfile.ZipFile(path,"r") as z:
        z.extractall(path = DOWNLOADED_DATA_DIRNAME + "/" + path[-9:-4])
    return DOWNLOADED_DATA_DIRNAME + "/" + path[-9:-4]

class BaseT91(Dataset):
  def __init__(self,data_path,train_transform=None,target_transform = None):
    super(T91Dataset,self).__init__()
    self.data_path =  extract_T91(data_path)  
    self.train_transform = train_transform
    self.target_transform = target_transform
    img_files = os.listdir(data_path)
    img_paths = [data_path + '/'+ img_file for img_file in img_files]
    self.img_paths = img_paths
    self.PIL_transform = Compose([ToPILImage()])

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self,idx):
    w = h = 33
    img = Image.open(self.img_paths[idx])
    img,_cb,_cr = img.convert('YCbCr').split()
    label_img = self.target_transform(img)
    train_img = self.PIL_transform(label_img)
    train_img = train_img.resize((int(w/3),int(h/3)))
    train_img = train_img.resize((w,h),Image.BICUBIC)
    train_img = self.train_transform(train_img)

    return train_img,label_img


class T91(BaseDataModule):
    '''
    Creating a T91 DataModule inheriting from the BaseDataModule class
    '''
    def __init__(self,args : argparse.Namespace) -> None:
        super().__init__(args)
        