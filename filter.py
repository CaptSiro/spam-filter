import os.path
import corpus
import tfidf
import utils
from tokenize import HTMLTokenStream
from email_reader import Email
from typing import Iterable, Tuple


class MyFilter:
    def __init__(self):
        tokenizer = HTMLTokenStream()
        self.analyzer = tfidf.WordCounter(tokenizer.stream)
        self.filter_vec = self.analyzer.tfidf_vec()


    @staticmethod
    def parse_emails(directory) -> Iterable[Tuple[str, Email]]:
        for file, _ in corpus.Corpus(directory).emails():
            with open(os.path.join(directory, file), "r", encoding="utf8") as fp:
                yield file, Email.from_file(fp)


    @staticmethod
    def is_ok(spec, item):
        if item not in spec:
            print(f"{item} not in spec")
            return True

        return spec[item] == "OK"


    @staticmethod
    def scal_mul(vec_x: dict, vec_y: dict):
        return sum((vec_x.get(term, 0) * vec_y.get(term, 0) for term in vec_x))


    def train(self, directory):
        spec = utils.read_classification_from_file(os.path.join(directory, "!truth.txt"))
        for file, email in MyFilter.parse_emails(directory):
            self.analyzer.documents.append(
                self.analyzer.scan(email.le_contante, MyFilter.is_ok(spec, file))
            )

        self.filter_vec = self.analyzer.tfidf_vec()


    def test(self, directory):
        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf8") as pp:
            for file, email in MyFilter.parse_emails(directory):
                email_vec = self.analyzer.vec(email.le_contante)
                pp.write(f"{file} {self.predict(email_vec)}\n")


    SPAM_THRESHOLD = -0.05
    def predict(self, email_vec: dict) -> str:
        word_freq = MyFilter.scal_mul(email_vec, self.filter_vec)
        result = word_freq

        return 'SPAM' if result < MyFilter.SPAM_THRESHOLD else 'OK'
