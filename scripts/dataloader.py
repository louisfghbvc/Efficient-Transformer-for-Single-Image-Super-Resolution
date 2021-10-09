from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

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

        Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
        Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]

        img_crop.append(img[:, Ws:Ws+self.crop_size, Hs:Hs+self.crop_size])
        label_crop.append(label[:, Ws*self.lr_scale:(Ws+self.crop_size) *
                                self.lr_scale, Hs*self.lr_scale: (Hs+self.crop_size)*self.lr_scale])
    

    return torch.stack(img_crop), torch.stack(label_crop)
