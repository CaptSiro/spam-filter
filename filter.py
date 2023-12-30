import math
import os.path
import corpus
import tfidf
import utils
from vectorize import Vectorizer
from tokenize import HTMLTokenStream
from email_reader import Email
from typing import Iterable, Tuple


class MyFilter:
    def __init__(self):
        tokenizer = HTMLTokenStream()
        self.analyzer = tfidf.WordCounter(tokenizer.stream)

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
        for file, email in corpus.Corpus(directory).parsed_emails():
            self.analyzer << self.analyzer.scan(email.le_contante, MyFilter.is_ok(spec, file))

        self.analyzer.save_state()

    def test(self, directory):
        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf8") as pp:
            for file, email in corpus.Corpus(directory).parsed_emails():
                pp.write(f"{file} {self.predict(file, email)}\n")


    SPAM_THRESHOLD = -0.05
    @staticmethod
    def calc(result):
        return 'SPAM' if result < MyFilter.SPAM_THRESHOLD else 'OK'


    def predict(self, file, email: "Email") -> str:
        word_freq = MyFilter.scal_mul(self.analyzer.vec(email.le_contante), self.analyzer.state_vec)
        result = word_freq
        return MyFilter.calc(result)



class MyFilterPropAnalysis:
    def __init__(self):
        tokenizer = HTMLTokenStream()
        self.analyzer = tfidf.WordCounter(tokenizer.stream)
        self.spam_vec = Vectorizer()
        self.ok_vec = Vectorizer()

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
        for file, email in corpus.Corpus(directory).parsed_emails():
            self.analyzer << self.analyzer.scan(email.le_contante, MyFilter.is_ok(spec, file))

            if MyFilter.is_ok(spec, file):
                self.ok_vec << Vectorizer.calc(email)
            else:
                self.spam_vec << Vectorizer.calc(email)

        self.analyzer.save_state()
        self.ok_vec.save_state()
        self.spam_vec.save_state()

    def test(self, directory):
        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf8") as pp:
            for file, email in corpus.Corpus(directory).parsed_emails():
                pp.write(f"{file} {self.predict(file, email)}\n")


    SPAM_THRESHOLD = 0.02
    @staticmethod
    def calc(result):
        return 'SPAM' if result < MyFilterPropAnalysis.SPAM_THRESHOLD else 'OK'


    def predict(self, file, email: "Email") -> str:
        word_freq = MyFilter.scal_mul(self.analyzer.vec(email.le_contante), self.analyzer.state_vec)

        email_vec = Vectorizer.calc(email)
        ok = self.ok_vec.distribute(email_vec).product()
        spam = self.spam_vec.distribute(email_vec).product()

        # division by 0
        ok = ok if ok != 0 else 1
        spam = spam if spam != 0 else 1

        result = word_freq
        before = MyFilter.calc(result)

        result = word_freq - math.log10(ok / spam)
        after = MyFilter.calc(result)

        # if before != after:
        #     print("%s\twf=%-8.2f os=%-8.2f = %s" % (file, word_freq, math.log10(ok / spam), after))

        return after



class NaiveBayesFilter:
    def __init__(self):
        tokenizer = HTMLTokenStream()
        self.spam = tfidf.WordCounter(tokenizer.stream)
        self.ok = tfidf.WordCounter(tokenizer.stream)


    def train(self, directory):
        spec = utils.read_classification_from_file(os.path.join(directory, "!truth.txt"))
        for file, email in corpus.Corpus(directory).parsed_emails():
            if MyFilter.is_ok(spec, file):
                self.ok << self.ok.scan(email.le_contante)
            else:
                self.spam << self.spam.scan(email.le_contante)

        self.ok.save_state()
        self.spam.save_state()


    def test(self, directory):
        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf8") as pp:
            for file, email in corpus.Corpus(directory).parsed_emails():
                pp.write(f"{file} {self.predict(email)}\n")


    def predict(self, email: Email):
        result_ok = MyFilter.scal_mul(self.ok.vec(email.le_contante), self.ok.state_vec)
        result_spam = MyFilter.scal_mul(self.spam.vec(email.le_contante), self.spam.state_vec)

        return 'SPAM' if result_spam > result_ok else 'OK'