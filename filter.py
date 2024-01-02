import math
import os.path
from sf_corpus import Corpus
import sf_tfidf
import utils
from sf_vectorize import Vectorizer
from sf_token import HTMLTokenStream
from sf_email_reader import Email
from sf_header_scan import SenderCounter



class MyFilter:
    KEY_WORD_FREQ = "word-frequency"
    KEY_SENDERS = "senders"
    KEY_PROP_ANALYSIS = "property-analysis"
    KEY_RESULT = "result"
    SPAM_THRESHOLD = -0.04

    def __init__(self):
        tokenizer = HTMLTokenStream()
        self.analyzer = sf_tfidf.WordCounter(tokenizer.stream)
        self.senders = SenderCounter()
        self.spam_vec = Vectorizer()
        self.ok_vec = Vectorizer()

        self.weights = {
            MyFilter.KEY_WORD_FREQ: 1,
            MyFilter.KEY_SENDERS: 0.5,
            MyFilter.KEY_PROP_ANALYSIS: 0.14
        }

    @staticmethod
    def is_ok(spec, item):
        if item not in spec:
            print(f"{item} not in spec")
            return True

        return spec[item] == "OK"

    @staticmethod
    def scal_mul(vec_x: dict, vec_y: dict):
        return sum((vec_x.get(term, 0) * vec_y.get(term, 0) for term in vec_x))

    def save_email(self, email, is_ok, weight=1):
        self.analyzer << self.analyzer.scan(email.le_contante, is_ok, weight=weight)
        self.senders.load_sender(email.headers, is_ok, weight)

        if is_ok:
            self.ok_vec << Vectorizer.calc(email, weight)
        else:
            self.spam_vec << Vectorizer.calc(email, weight)

    @staticmethod
    def adjust_instance(score: float):
        return 1 + abs(score) * 0.35

    def adjust_group(self, group: str, score: float, is_reward: bool):
        if is_reward:
            self.weights[group] += abs(score) * 0.1 * self.weights[group]
        else:
            self.weights[group] -= abs(score) * 0.18 * self.weights[group]

    def create_adjustment(self, values, expected):
        def perform(group, save_item):
            score = values[group]
            if MyFilter.calc(score, 0) != expected:
                save_item(MyFilter.adjust_instance(score))
                self.adjust_group(group, score, False)
            else:
                save_item(1)
                self.adjust_group(group, score, True)

        return perform

    def validate_training(self, corpus, validation, spec):
        correct = 0

        self.analyzer.save_state()
        self.ok_vec.save_state()
        self.spam_vec.save_state()

        for file, email in corpus.parse_partition(validation):
            prediction, values = self.predict(file, email)

            is_ok = MyFilter.is_ok(spec, file)
            expected = spec[file]

            if prediction == expected:
                self.save_email(email, is_ok)
                correct += 1
                continue

            adjust = self.create_adjustment(values, expected)
            adjust(
                MyFilter.KEY_WORD_FREQ,
                lambda w: self.analyzer << self.analyzer.scan(email.le_contante, is_ok, w)
            )

            adjust(
                MyFilter.KEY_SENDERS,
                lambda w: self.senders.load_sender(email.headers, is_ok, w)
            )

            def adjust_prop_analysis(weight):
                if is_ok:
                    self.ok_vec << Vectorizer.calc(email, weight)
                else:
                    self.spam_vec << Vectorizer.calc(email, weight)

            adjust(
                MyFilter.KEY_PROP_ANALYSIS,
                adjust_prop_analysis
            )

        return correct / len(validation)

    def train(self, directory):
        spec = utils.read_classification_from_file(os.path.join(directory, "!truth.txt"))
        corpus = Corpus(directory)

        training, validation = corpus.partitions()
        for file, email in corpus.parse_partition(training):
            is_ok = MyFilter.is_ok(spec, file)
            self.save_email(email, is_ok)

        for _ in range(10):
            percentage = self.validate_training(corpus, validation, spec)
            print(percentage)
            if percentage > 0.96:
                break

        print(self.weights)

    def test(self, directory):
        self.analyzer.save_state()
        self.ok_vec.save_state()
        self.spam_vec.save_state()

        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf-8") as pp:
            for file, email in Corpus(directory).parsed_emails():
                prediction, _ = self.predict(file, email)
                pp.write(f"{file} {prediction}\n")


    @staticmethod
    def calc(result, threshold=None):
        if threshold is None:
            return 'SPAM' if result < MyFilter.SPAM_THRESHOLD else 'OK'

        return 'SPAM' if result < threshold else 'OK'

    def predict(self, file, email: "Email") -> (str, dict[str, float]):
        word_freq = MyFilter.scal_mul(self.analyzer.vec(email.le_contante), self.analyzer.state_vec) \
                    * self.weights[MyFilter.KEY_WORD_FREQ]

        senders = self.senders.test_sender(email.headers) \
                  * self.weights[MyFilter.KEY_SENDERS]

        email_vec = Vectorizer.calc(email)
        ok = self.ok_vec.distribute(email_vec).sumlog()
        spam = self.spam_vec.distribute(email_vec).sumlog()
        # division by 0
        ok = ok if ok != 0 else 1
        spam = spam if spam != 0 else 1
        prop_analysis = -(math.log10(ok / spam)) * self.weights[MyFilter.KEY_PROP_ANALYSIS]

        result = word_freq + senders + prop_analysis

        return MyFilter.calc(result), {
            MyFilter.KEY_WORD_FREQ: word_freq,
            MyFilter.KEY_SENDERS: senders,
            MyFilter.KEY_PROP_ANALYSIS: prop_analysis,
            MyFilter.KEY_RESULT: result
        }