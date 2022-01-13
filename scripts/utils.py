import typing
from pathlib import Path
import re
from typing import Optional
from .logger import *
import torch
import numpy as np
def _copy(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)

Path.copy = _copy

DEBUG = False

if DEBUG:
    warn("DEBUG mode is on, if you are not DEBUG please disable or model can't be trained")
class GlobalPathes:
    def __init__(self):
        self.ROOT = Path(".")
        self.MODELS = Path(self.ROOT, "models").resolve()
        self.SCRIPTS = Path(self.ROOT, "scripts").resolve()
        self.CONFIGS = Path(self.ROOT, "configs").resolve()
        self.SHELLS =  Path(self.ROOT, "shells").resolve()
        self.CHECKPOINTS = Path(self.ROOT, "checkpoints").resolve() 
        self.DATASETS = Path(self.ROOT, "datasets").resolve() 
        self.CONFIG_DEFAULT = Path(self.CONFIGS, "default.yaml")
        self.CONFIG_OVERRIDE = Path(self.CONFIGS, "override.yaml")
        
    

PATHS = GlobalPathes()
def mkdir(p):
    p.mkdir(exist_ok = True, parents = True)
def mkdirs(plist):
    for p in plist:
        mkdir(p)
def createFile(p):
    p.parent.mkdir(exist_ok = True, parents = True)
    if not p.exists():
        p.open("w+")
def createFiles(plist):
    for p in plist:
        createFile(p)
def getFiles(p,pattern="*"):
    return [p for p in Path(p).glob(pattern)]
def getFilesr(p,pattern="*"):
    return [p for p in Path(p).rglob(pattern)]
def getFile(p,pattern="*"):
    for p in Path(p).glob(pattern): return p
def getFiler(p,pattern="*"):
    for p in Path(p).rglob(pattern): return p
def getFinalEpoch(checkpointFolder):  # return last epoch num (final training saved)
    p = Path(checkpointFolder)
    if not p.exists():
        return 0
    files = getFiles(p, "epoch*.pth")
    nums = sorted([int(re.match(r'epoch(\d+)', x.stem).group(1)) for x in files])
    if len(nums) == 0 :
        return 0
    return nums[-1]
def getExistPath(paths):
    for p in paths:
        if type(p) == str:
            p = Path(p)
        if p.exists():
            return p
    return None



## src = https://github.com/yjn870/SRCNN-pytorch/blob/master/utils.py


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class Grad:
    def __enter__(self):
        ...
    def __exit__(self, type, value, traceback):
        ...





