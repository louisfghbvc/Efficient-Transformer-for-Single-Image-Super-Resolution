import typing
from pathlib import Path
import re
from typing import Optional
from .logger import *

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
    nums = [int(re.match(r'epoch(\d+)', x.stem).group(1)) for x in files]
    if len(nums) == 0 :
        return 0
    return nums[-1]




    
class Grad:
    def __enter__(self):
        ...
    def __exit__(self, type, value, traceback):
        ...





