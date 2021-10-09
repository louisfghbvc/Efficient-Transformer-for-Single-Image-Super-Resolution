from fire import Fire
from scripts.efficientTransformerSR import EfficientTransformerSR
def main():
    app = EfficientTransformerSR()
    app.train()
if __name__ == '__main__':
    Fire(main)