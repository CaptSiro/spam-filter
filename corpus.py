import os
from pathlib import Path


class Corpus:
    def __init__(self, path):
        self.path = path

    def emails(self):
        for file in os.listdir(self.path):
            if file[0] != '!':
                yield file, Path(os.path.join(self.path, file)).read_text(encoding="utf-8")
