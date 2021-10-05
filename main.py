from fire import Fire
from scripts.efficientTransformerSR import EfficientTransformerSR
def main(config=""):
    app = EfficientTransformerSR(config)
    app.train()
if __name__ == '__main__':
    Fire(main)