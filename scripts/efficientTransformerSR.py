from pathlib import Path

import torch
import torch.optim as optim
import tqdm

from torch.utils.data import DataLoader
from models.models import ESRT
from .utils import *
from .configParser import ConfigParser

class EfficientTransformerSR:
    def __init__(self, configs= "train"):
        self.configs = None
        self.initConfigs(configs)

    def initConfigs(self, configs ):
        self.configs = configs or self.configs
        self.configs = ConfigParser(self.configs).content
        mkdirs([PATHS.MODELS, PATHS.SCRIPTS, PATHS.SCRIPTS, PATHS.CONFIGS, PATHS.SHELLS])
        createFiles([PATHS.CONFIG_DEFAULT, PATHS.CONFIG_OVERRIDE])
        if self.configs["usegpu"] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
            print('Warning! Using CPU.')

    def epochsAction(self, action , start, end, loader):
        for epoch in range(start, end):    
            self.epochAction(action, loader)
            self.epoch = epoch
            self.endEpoch = end
    def forwardAction(self, x, y):
        device = self.device
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
                totalLoss += loss 
                totalCorrect += torch.sum(y == out)
                if isBackward:
                    loss.backward()
                    self.optimizer.step()
                epochProgress = epochProgress if action == "test" else "1/1"
                
                batchLoader.set_description(desc=f"[{epochProgress}] {action} loss : {round(loss.item() / len(y),2)} ")
    
    def train(self):
        self.epochsAction("train",self.trainloader)
    def valid(self):
        self.epochAction("valid",self.validloader)
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
        return PATHS.CHECKPOINTS / f"ETSR-lr{ self.configs['learningRate'] }-flip{self.configs['randomFlip']}-psize{self.configs['patchSize']}" 
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
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = ESRT()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlatea(self.optimizer)


    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

if __name__ == '__main__':
    a = EfficientTransformerSR("train")


