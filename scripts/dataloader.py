from torch.utils.data import Dataset
from PIL import Image
from .utils import PATHS, getFile, getFiler, getFiles, getFilesr, getExistPath
from .logger import *
import numpy as np
import torch
from torchvision import transforms
class DIV2KDataset(Dataset):
  def __init__(self, root_dir, crop_size=48, rotation=False, transform=None, is_training=True, lr_scale = 2):
    '''
    Args:
      root_dir: Directory with all the images
      crop_size: crop image size, if no crop, then input = -1
      transform: transform data based on torchvision transform function
      is_training: (True) - training dataset, (False) - validation dataset
      lr_scale: low resolution image scale
    '''
    self.root_dir = root_dir
    self.is_training = is_training
    self.crop_size = crop_size
    self.rotation = rotation
    self.transform = transform
    self.lr_scale = lr_scale
    root_dir = str(root_dir.resolve().absolute())
    self.train_hr_image_path = root_dir + '/train/HR/X1'
    self.train_lr_image_path = root_dir + '/train/LR/X' + str(lr_scale)
    self.valid_hr_image_path = root_dir + '/val/HR/X1'
    self.valid_lr_image_path = root_dir + '/val/LR/X' + str(lr_scale) 


    
  
  def __len__(self):
    return 800 if self.is_training else 100
  
  def __getitem__(self, idx):
    img_path = ""
    if self.is_training:
      img_path = self.train_lr_image_path + '/' + \
          str(idx + 1).zfill(4) + 'x' + str(self.lr_scale) + '.png'
      label_path = self.train_hr_image_path + \
          '/' + str(idx + 1).zfill(4) + '.png'
    else:
      img_path = self.valid_lr_image_path + '/' + \
          str(idx + 801).zfill(4) + 'x' + str(self.lr_scale) + '.png'
      label_path = self.valid_hr_image_path + \
          '/' + str(idx + 801).zfill(4) + '.png'

    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path).convert('RGB')

    if self.rotation:
      pass
    
    if self.transform:
      img = self.transform(img)
      label = self.transform(label)

    img_crop = []
    label_crop = []
    if self.crop_size != -1:
      for i in range(16):
        W = img.size()[1]
        H = img.size()[2]

        Ws = np.random.randint(0, W-self.crop_size+1, 1)[0]
        Hs = np.random.randint(0, H-self.crop_size+1, 1)[0]

        img_crop.append(img[:, Ws:Ws+self.crop_size, Hs:Hs+self.crop_size])
        label_crop.append(label[:, Ws*self.lr_scale:(Ws+self.crop_size) *
                                self.lr_scale, Hs*self.lr_scale: (Hs+self.crop_size)*self.lr_scale])
    

    return torch.stack(img_crop), torch.stack(label_crop)


class TestLoader(Dataset):
  def __init__(self, root_dir, crop_size=48, rotation=False, transform=None, lr_scale = 2):
    '''
    Args:
      root_dir: Directory with all the images
      crop_size: crop image size, if no crop, then input = -1
      transform: transform data based on torchvision transform function
      is_training: (True) - training dataset, (False) - validation dataset
      lr_scale: low resolution image scale
    '''
    self.root_dir = root_dir = getExistPath([root_dir, PATHS.DATASETS / root_dir]) # root dir absolut path | datasetPath/yourdatset under this project
    if not root_dir:
      error(f"path not found {str(root_dir)}")
      return 
    self.dataset_path = dataset_path = getFile(root_dir, f"image_SRF_{lr_scale}")
    if not dataset_path:
      error(f"x {lr_scale} scale path not found under {str(dataset_path)}")
      return 
    self.len = len(getFiles(dataset_path, "*LR*.png"))
    self.genItemDic()
    self.dataset_name = root_dir.stem 
    self.crop_size = crop_size
    self.rotation = rotation
    self.transform = transform
    self.lr_scale = lr_scale
  def getNumFromPath(self,p):
    return int(str(p.stem or p)[4:7])
  def genItemDic(self):
    self.LRPaths = lrp = sorted(getFiles(self.dataset_path, "*LR*.png"))
    self.HRPaths = hrp = sorted(getFiles(self.dataset_path, "*HR*.png"))
    self.items = []
    
    
    li, hi = 0, 0
    while li < len(lrp) and hi < len(hrp):
      if self.getNumFromPath(lrp[li]) == self.getNumFromPath(hrp[hi]):
        self.items.append([lrp[li], hrp[hi]])
        li+=1
        hi+=1
      elif self.getNumFromPath(lrp[li]) < self.getNumFromPath(hrp[hi]):
        warn(f"lost HR file of {lrp[li].stem}")
        li+=1
      elif self.getNumFromPath(lrp[li]) > self.getNumFromPath(hrp[hi]):
        warn(f"lost LR file of {hrp[hi].stem}")
        hi+=1
      
      
      

  def __len__(self):
    self.len
  def __getitem__(self, idx):
    img_path, label_path = self.items[idx]

    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path).convert('RGB')

    # if self.rotation:
    #   pass
    
    if self.transform:
      img = self.transform(img)
      label = self.transform(label)

    # img_crop = []
    # label_crop = []
    # if self.crop_size != -1:
    #   for i in range(16):
    #     W = img.size()[1]
    #     H = img.size()[2]

    #     Ws = np.random.randint(0, W-self.crop_size+1, 1)[0]
    #     Hs = np.random.randint(0, H-self.crop_size+1, 1)[0]

        # img_crop.append(img[:, Ws:Ws+self.crop_size, Hs:Hs+self.crop_size])
        # label_crop.append(label[:, Ws*self.lr_scale:(Ws+self.crop_size) *
        #                         self.lr_scale, Hs*self.lr_scale: (Hs+self.crop_size)*self.lr_scale])
    

    return transforms.ToTensor()(img).unsqueeze(0).unsqueeze(0), transforms.ToTensor()(label).unsqueeze(0).unsqueeze(0)
