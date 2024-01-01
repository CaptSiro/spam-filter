import math
import os.path
import corpus
import tfidf
import utils
from vectorize import Vectorizer
from tokenize import HTMLTokenStream
from email_reader import Email
from header_scan import SenderCounter
from typing import Iterable, Tuple


class MyFilter:
    def __init__(self):
        tokenizer = HTMLTokenStream()
        self.analyzer = tfidf.WordCounter(tokenizer.stream)
        self.senders = SenderCounter()

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
            self.senders.load_sender(email.headers, MyFilter.is_ok(spec, file))

        self.analyzer.save_state()

    def test(self, directory):
        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf8") as pp:
            for file, email in corpus.Corpus(directory).parsed_emails():
                pp.write(f"{file} {self.predict(file, email)}\n")


    SPAM_THRESHOLD = -0.05
    @staticmethod
    def calc(result):
        return 'SPAM' if result < MyFilter.SPAM_THRESHOLD else 'OK'

    SENDER_WEIGHT = 0.5
    def predict(self, file, email: "Email") -> str:
        word_freq = MyFilter.scal_mul(self.analyzer.vec(email.le_contante), self.analyzer.state_vec)
        result = word_freq
        result += self.senders.test_sender(email.headers) * 0.5
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


    SPAM_THRESHOLD = -0.05
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

        property_analysis = -(math.log10(ok / spam) * 0.14)

        result = word_freq
        before = MyFilterPropAnalysis.calc(result)

        result = word_freq + property_analysis
        after = MyFilterPropAnalysis.calc(result)

        # if before != after:
        #     print("%s\t%-8.2f %s %-8.3f = %-8.4f\t%s" % (file, word_freq, '+' if 1 == math.copysign(1, property_analysis) else '-', abs(property_analysis), result, after))

        return after



class MyFilterPropAnalysisOnly:
    def __init__(self):
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
            if MyFilter.is_ok(spec, file):
                self.ok_vec << Vectorizer.calc(email)
            else:
                self.spam_vec << Vectorizer.calc(email)

        self.ok_vec.save_state()
        self.spam_vec.save_state()

    def test(self, directory):
        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf8") as pp:
            for file, email in corpus.Corpus(directory).parsed_emails():
                pp.write(f"{file} {self.predict(file, email)}\n")


    SPAM_THRESHOLD = 0.02
    @staticmethod
    def calc(result):
        return 'SPAM' if result < MyFilterPropAnalysisOnly.SPAM_THRESHOLD else 'OK'


    def predict(self, file, email: "Email") -> str:
        email_vec = Vectorizer.calc(email)
        ok = self.ok_vec.distribute(email_vec).product()
        spam = self.spam_vec.distribute(email_vec).product()

        return 'SPAM' if spam > ok else 'OK'



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
