from fire import Fire
from scripts.efficientTransformerSR import EfficientTransformerSR
def main(config = "train"):
    app = EfficientTransformerSR(config)
    app.train()
    #app.test()
if __name__ == '__main__':
    Fire(main)