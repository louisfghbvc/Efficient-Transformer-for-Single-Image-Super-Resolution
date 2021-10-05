import typing
from pathlib import Path
import re
from typing import Optional
class GlobalPathes:
    def __init__(self):
        self.ROOT = Path("D:\Efficient-Transformer-for-Single-Image-Super-Resolution")
        self.MODELS = Path(self.ROOT, "models").resolve()
        self.SCRIPTS = Path(self.ROOT, "scripts").resolve()
        self.CONFIGS = Path(self.ROOT, "configs").resolve()
        self.SHELLS =  Path(self.ROOT, "shells").resolve()
        self.CHECKPOINTS = Path(self.ROOT, "checkpoints").resolve() 
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
        return None
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




