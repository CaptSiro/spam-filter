import math
import re
import email_reader
import tokenize
import utils


class Vectorizer:
    class Vec:
        PROP_COUNT = 3
        def __init__(self, props):
            self.props = props
            if len(self.props) != self.PROP_COUNT:
                raise ValueError(f"Props must have {self.PROP_COUNT} items")

        def product(self):
            return abs(sum((math.log10(p) for p in self.props if p != 0)))

        def __str__(self):
            return str(self.props)

    html_content = tokenize.HTMLContent()
    link_counter = tokenize.HTMLLinkCounter()
    price_re = re.compile(r"($[0-9,]+)")

    def __init__(self):
        self.vectors = []
        self.avg = self.avg_vec()
        self.stddev = self.stddev_vec(self.avg)

    def __lshift__(self, other):
        self.vectors.append(other)

    def save_state(self):
        self.avg = self.avg_vec()
        self.stddev = self.stddev_vec(self.avg)

    def avg_vec(self) -> "Vectorizer.Vec":
        if len(self.vectors) == 0:
            return Vectorizer.Vec([0 for _ in range(Vectorizer.Vec.PROP_COUNT)])

        sums = [sum((self.vectors[r].props[i]) for r in range(len(self.vectors))) for i in range(Vectorizer.Vec.PROP_COUNT)]
        return Vectorizer.Vec([sums[i] / len(self.vectors) for i in range(Vectorizer.Vec.PROP_COUNT)])

    def stddev_vec(self, avg_vec: "Vectorizer.Vec") -> "Vectorizer.Vec":
        if len(self.vectors) == 0:
            return Vectorizer.Vec([0 for _ in range(Vectorizer.Vec.PROP_COUNT)])

        sums = [sum(((self.vectors[r].props[i] - avg_vec.props[i]) ** 2 for r in range(len(self.vectors)))) for i in range(Vectorizer.Vec.PROP_COUNT)]
        return Vectorizer.Vec([(sums[i] / len(self.vectors)) ** 0.5 for i in range(Vectorizer.Vec.PROP_COUNT)])

    def distribute(self, vec: "Vectorizer.Vec") -> "Vectorizer.Vec":
        return Vectorizer.Vec([utils.normal_dist(self.avg.props[i], self.stddev.props[i], vec.props[i]) for i in range(Vectorizer.Vec.PROP_COUNT)])

    @staticmethod
    def calc(email: email_reader.Email) -> "Vectorizer.Vec":
        content = " ".join(Vectorizer.html_content.extract(email.le_contante))
        capitalised = sum((len(word) for word in tokenize.t(content, lambda w: str(w).upper() == str(w))))
        # length = len(content)
        links = Vectorizer.link_counter.count(content)
        exclamations = content.count('!')

        return Vectorizer.Vec([capitalised, links, exclamations])
