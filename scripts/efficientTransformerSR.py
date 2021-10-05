from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader
from models.models import ESRT
from .utils import *
from .configParser import ConfigParser

class EfficientTransformerSR:
    def __init__(self, configs= "train"):
        self.configs = None
        self.epoch = None
        self.initConfigs(configs)
        self.initParams()

    def initConfigs(self, configs ):
        self.configs = configs or self.configs
        self.configs = ConfigParser(self.configs).content
        mkdirs([PATHS.MODELS, PATHS.SCRIPTS, PATHS.SCRIPTS, PATHS.CONFIGS, PATHS.SHELLS, PATHS.CHECKPOINTS, PATHS.DATASETS])
        createFiles([PATHS.CONFIG_DEFAULT, PATHS.CONFIG_OVERRIDE])
        if self.configs["usegpu"] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
            warn('Using CPU.')
        

    def trainEpochs(self, start, end):
        self.epoch = start
        self.endEpoch = end
        for epoch in range(start, end):    
            self.epoch = epoch
            self.epochAction("train", self.trainloader)
            self.epochAction("valid", self.validloader)
            
    def forwardAction(self, x, y):
        device = self.device
        if DEBUG: return torch.Tensor([[1]]), torch.Tensor([0])
        x, y = x.to(device), y.to(device)
        out = self.model(x)
        loss = self.criterion(out, y)
        return out, loss

    def epochAction(self, action, loader):
        isBackward = True if action == "train" else False
        GradSelection = Grad if isBackward else torch.no_grad
        totalLoss, totalCorrect = 0, 0
        batchLoader = tqdm(loader)
        with GradSelection():
            for x, y in batchLoader:
                self.optimizer.zero_grad()
                out, loss = self.forwardAction(x,y)
                if not DEBUG:
                    totalLoss += loss 
                    totalCorrect += torch.sum(y == out)
                    if isBackward:
                        loss.backward()
                        self.optimizer.step()
                epochProgress = f"{self.epoch+1}/{self.configs['epochs']}" if action != "test" else "1/1"
                
                batchLoader.set_description(desc=f"[{epochProgress}] {action} loss : {round(loss.item() / len(y),2)} ")
    
    def train(self, loader = None):
        self.trainloader = loader or self.trainloader
        self.load()
        self.trainEpochs( self.startEpoch, self.configs["epochs"])
    def valid(self, loader = None):
        loader = loader or self.validloader
        self.epochAction("valid",loader)
    def test(self):
        ...
    def saveObject(self, epoch):
        return {
            "epoch" : epoch,
            "model" : self.model.state_dict(),
            "scheduler" : self.scheduler.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
    def getCheckpointFolder(self):
        return PATHS.CHECKPOINTS / f"ETSR-lr{ self.configs['startLearningRate'] }-flip{self.configs['randomFlip']}-psize{self.configs['patchSize']}" 
    def save(self,fileName, epoch=1):
        fileName = fileName or f"epoch{epoch}.pth"
        saveFolder = self.getCheckpointFolder()
        mkdir(saveFolder)
        fileName = saveFolder / fileName
        torch.save(self.saveObject(epoch), fileName)
    def load(self):
        saveFolder = self.getCheckpointFolder()
        startEpoch = self.configs["startEpoch"]
        
        startEpoch = getFinalEpoch(saveFolder) if startEpoch == -1 else startEpoch # get real last epoch if -1
        self.startEpoch = startEpoch
        if startEpoch == 0:
            return #if 0 no load (including can't find )
        
        modelFile = getFile(saveFolder, "epoch{startEpoch}")
        saveObject = torch.load(modelFile)
        self.model.load_state_dict(saveObject["model"])
        self.scheduler.load_state_dict(saveObject["scheduler"])
        self.optimizer.load_stat_dict(saveObject["optimizer"])
        
    def loadParams(self):
        ...
    def initParams(self):
        self.criterion = torch.nn.L1Loss()
        self.model = ESRT()
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["startLearningRate"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer )
        #(TODO) : implement loader 
        self.trainloader = [(f"train x{i}","y{i}") for i in range(1000)] 
        self.validloader = [(f"valid x{i}","y{i}") for i in range(500)]


if __name__ == '__main__':
    a = EfficientTransformerSR("train")


