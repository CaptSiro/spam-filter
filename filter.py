import math
import os.path

import sf_counter
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
            if percentage > 0.96:
                break

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




class BayesFilter:
    KEY_WORD_FREQ = "word-frequency"
    KEY_SENDERS = "senders"
    KEY_SUBJECT = "subject"
    KEY_PROP_ANALYSIS = "property-analysis"
    KEY_RESULT = "result"
    SPAM_THRESHOLD = -0.04

    def __init__(self):
        tokenizer = HTMLTokenStream()

        dataset_content = set()
        self.analyzer_ok = sf_counter.WordCounter(tokenizer.stream, dataset_content)
        self.analyzer_spam = sf_counter.WordCounter(tokenizer.stream, dataset_content)
        dataset_subject = set()
        self.subject_ok = sf_counter.WordCounter(tokenizer.stream, dataset_subject)
        self.subject_spam = sf_counter.WordCounter(tokenizer.stream, dataset_subject)

        self.senders_ok = SenderCounter()
        self.senders_spam = SenderCounter()
        self.vec_spam = Vectorizer()
        self.vec_ok = Vectorizer()
        self.types = sf_counter.UniqueCounter()

        self.weights = {
            BayesFilter.KEY_WORD_FREQ: (1, 1),
            BayesFilter.KEY_SENDERS: (0.5, 0,5),
            BayesFilter.KEY_PROP_ANALYSIS: (0.14, 0.14)
        }

    def save_state(self):
        self.vec_spam.save_state()
        self.vec_ok.save_state()

    @staticmethod
    def is_ok(spec, item):
        if item not in spec:
            print(f"{item} not in spec")
            return True

        return spec[item] == "OK"

    @staticmethod
    def scal_mul(vec_x: dict, vec_y: dict):
        return sum((vec_x.get(term, 0) * vec_y.get(term, 0) for term in vec_x))

    def save_email(self, email: Email, is_ok, weight=1):
        if is_ok:
            self.vec_ok << Vectorizer.calc(email, weight)
            self.analyzer_ok << self.analyzer_ok.scan(email.le_contante)
            self.subject_ok << self.subject_ok.scan(email.headers["subject"])
            self.senders_ok.load_sender(email.headers, True, weight)
        else:
            self.vec_spam << Vectorizer.calc(email, weight)
            self.subject_spam << self.subject_spam.scan(email.headers["subject"])
            self.analyzer_spam << self.analyzer_spam.scan(email.le_contante)
            self.senders_spam.load_sender(email.headers, True, weight)

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
            if BayesFilter.calc(score, 0) != expected:
                save_item(BayesFilter.adjust_instance(score))
                self.adjust_group(group, score, False)
            else:
                save_item(1)
                self.adjust_group(group, score, True)

        return perform

    def validate_training(self, corpus, validation, spec):
        correct = 0
        self.save_state()

        for file, email in corpus.parse_partition(validation):
            prediction, values = self.predict(file, email)

            is_ok = BayesFilter.is_ok(spec, file)
            expected = spec[file]

            if prediction == expected:
                self.save_email(email, is_ok)
                correct += 1
                continue

            def adjust_split(ok, spam):
                def perform(weight):
                    if is_ok:
                        ok(weight)
                    else:
                        spam(weight)

                return perform

            adjust = self.create_adjustment(values, expected)
            adjust(
                BayesFilter.KEY_WORD_FREQ,
                adjust_split(
                    lambda w: self.analyzer_ok << self.analyzer_ok.scan(email.le_contante, True, w),
                    lambda w: self.analyzer_spam << self.analyzer_spam.scan(email.le_contante, True, w),
                )
            )

            adjust(
                BayesFilter.KEY_SENDERS,
                adjust_split(
                    lambda w: self.senders_ok.load_sender(email.headers, True, w),
                    lambda w: self.senders_spam.load_sender(email.headers, True, w),
                )
            )

            adjust(
                BayesFilter.KEY_PROP_ANALYSIS,
                adjust_split(
                    lambda w: self.vec_ok << Vectorizer.calc(email, w),
                    lambda w: self.vec_spam << Vectorizer.calc(email, w),
                )
            )

        return correct / len(validation)

    def print_predict_parts(self, values):
        p_init_ok = math.log10(self.types.get("OK") / self.types.total)
        p_init_spam = math.log10(self.types.get("SPAM") / self.types.total)

        p_wf_ok, p_wf_spam = values[BayesFilter.KEY_WORD_FREQ]
        p_prop_ok, p_prop_spam = values[BayesFilter.KEY_PROP_ANALYSIS]
        p_subject_ok, p_subject_spam = values[BayesFilter.KEY_SUBJECT]
        result_ok, result_spam = values[BayesFilter.KEY_RESULT]

        print("\tP(OK)   = %-8.3f + %-8.3f + %-8.3f + %-8.3f = %-8.3f" % (p_init_ok, p_wf_ok, p_subject_ok, p_prop_ok, result_ok))
        print("\tP(SPAM) = %-8.3f + %-8.3f + %-8.3f + %-8.3f = %-8.3f" % (p_init_spam, p_wf_spam, p_subject_spam, p_prop_spam, result_spam))

    def train(self, directory):
        spec = utils.read_classification_from_file(os.path.join(directory, "!truth.txt"))
        corpus = Corpus(directory)

        training, validation = corpus.partitions()
        for file, email in corpus.parse_partition(training):
            is_ok = BayesFilter.is_ok(spec, file)
            self.save_email(email, is_ok)
            self.types.scan(spec[file])

        self.save_state()
        for file, email in corpus.parse_partition(validation):
            print(file)
            prediction, values = self.predict(file, email)
            self.print_predict_parts(values)

        # for _ in range(10):
        #     percentage = self.validate_training(corpus, validation, spec)
        #     if percentage > 0.96:
        #         break

    def test(self, directory):
        self.save_state()

        with open(os.path.join(directory, "!prediction.txt"), "w", encoding="utf-8") as pp:
            for file, email in Corpus(directory).parsed_emails():
                prediction, _ = self.predict(file, email)
                pp.write(f"{file} {prediction}\n")

    @staticmethod
    def calc(result, threshold=None):
        if threshold is None:
            return 'SPAM' if result < BayesFilter.SPAM_THRESHOLD else 'OK'

        return 'SPAM' if result < threshold else 'OK'

    def predict(self, file, email: "Email") -> (str, dict[str, float]):
        p_init_spam = math.log10(self.types.get("SPAM") / self.types.total)
        p_init_ok = math.log10(self.types.get("OK") / self.types.total)

        document = self.analyzer_ok.scan(email.le_contante)

        p_wf_ok = -math.pow(abs(self.analyzer_ok.probability(document)), 0.9)
        p_wf_spam = -math.pow(abs(self.analyzer_spam.probability(document)), 0.9)

        subject = self.subject_ok.scan(email.headers["subject"])
        p_subject_ok = self.subject_ok.probability(subject)
        p_subject_spam = self.subject_spam.probability(subject)

        # todo do senders later
        # senders = self.senders.test_sender(email.headers) * self.weights[BayesFilter.KEY_SENDERS]

        email_vec = Vectorizer.calc(email)

        p_prop_ok = self.vec_ok.distribute(email_vec).sumlog()
        p_prop_spam = self.vec_spam.distribute(email_vec).sumlog()

        # if p_wf_ok == 0:
        #     print(self.analyzer_ok.vec(email.le_contante).keys(), self.analyzer_ok.state_vec.keys())
        # if p_wf_spam == 0:
        #     print(self.analyzer_spam.vec(email.le_contante).keys(), self.analyzer_spam.state_vec.keys())

        result_ok = p_init_ok + p_wf_ok + p_subject_ok + p_prop_ok
        result_spam = p_init_spam + p_wf_spam + p_subject_spam + p_prop_spam

        return "SPAM" if result_spam > result_ok else "OK", {
            BayesFilter.KEY_WORD_FREQ: (p_wf_ok, p_wf_spam),
            BayesFilter.KEY_SENDERS: (0.5, 0.5),
            BayesFilter.KEY_SUBJECT: (p_subject_ok, p_subject_spam),
            BayesFilter.KEY_PROP_ANALYSIS: (p_prop_ok, p_prop_spam),
            BayesFilter.KEY_RESULT: (result_ok, result_spam)
        }
