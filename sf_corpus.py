import os
import shutil
from pathlib import Path
import utils
from sf_email_reader import Email


class Corpus:
    def __init__(self, path):
        self.path = path

    def emails(self):
        for file in os.listdir(self.path):
            if file[0] != '!':
                yield file, Path(os.path.join(self.path, file)).read_text(encoding="utf-8")

    def parsed_emails(self):
        for file in os.listdir(self.path):
            if file[0] != '!':
                with open(os.path.join(self.path, file), "r", encoding="utf-8") as fp:
                    yield file, Email.from_file(fp)

    def partitions(self):
        files = [f for f in os.listdir(self.path) if f[0] != '!']
        split_at = len(files)
        return []

    @staticmethod
    def copy_file_slice(partition, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)

        os.mkdir(directory)

        with open(os.path.join(directory, "!truth.txt"), "w", encoding="utf-8") as truth:
            for file, label in partition:
                shutil.copyfile(os.path.join(directory, "..", "emails", file), os.path.join(directory, file))
                truth.write(f"{file} {label}\n")

    @staticmethod
    def random_corpus(corpus_dir: str):
        emails = [*utils.read_classification_from_file(os.path.join(corpus_dir, "metadata.txt")).items()]
        utils.array_shuffle(emails)

        split_at = len(emails) // 2

        Corpus.copy_file_slice(emails[:split_at], os.path.join(corpus_dir, "train"))
        Corpus.copy_file_slice(emails[split_at:], os.path.join(corpus_dir, "test"))
