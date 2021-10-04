from pathlib import Path
class GlobalPathes:
    def __init__(self) -> None:
        self.root = Path("..")
        self.models = Path(self.root, "models").resolve()
        self.scripts = Path(self.root, "scripts").resolve()
        self.configs = Path(self.root, "configs").resolve()
        self.shell =  Path(self.root, "shells").resolve()

Paths = GlobalPathes()

