import math
import re
from sf_email_reader import Email
import sf_token
import utils


class Vectorizer:
    class Vec:
        PROP_COUNT = 7
        def __init__(self, props, weight):
            self.props = props
            self.weight = weight
            if len(self.props) != self.PROP_COUNT:
                raise ValueError(f"Props must have {self.PROP_COUNT} items")

        def sumlog(self):
            return sum((math.log10(p) for p in self.props if p != 0))

        def __str__(self):
            return str(self.props)

    html_content = sf_token.HTMLContent()
    link_counter = sf_token.HTMLLinkCounter()
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
            return Vectorizer.Vec([0 for _ in range(Vectorizer.Vec.PROP_COUNT)], 1)

        sums = [sum((self.vectors[r].props[i]) for r in range(len(self.vectors))) for i in range(Vectorizer.Vec.PROP_COUNT)]
        weight_sum = sum((v.weight for v in self.vectors))
        return Vectorizer.Vec([sums[i] / weight_sum for i in range(Vectorizer.Vec.PROP_COUNT)], 1)

    def stddev_vec(self, avg_vec: "Vectorizer.Vec") -> "Vectorizer.Vec":
        if len(self.vectors) == 0:
            return Vectorizer.Vec([0 for _ in range(Vectorizer.Vec.PROP_COUNT)], 1)

        sums = [sum(((self.vectors[r].props[i] - avg_vec.props[i]) ** 2 for r in range(len(self.vectors)))) for i in range(Vectorizer.Vec.PROP_COUNT)]
        return Vectorizer.Vec([(sums[i] / len(self.vectors)) ** 0.5 for i in range(Vectorizer.Vec.PROP_COUNT)], 1)

    def distribute(self, vec: "Vectorizer.Vec") -> "Vectorizer.Vec":
        return Vectorizer.Vec([utils.normal_dist(self.avg.props[i], self.stddev.props[i], vec.props[i]) for i in range(Vectorizer.Vec.PROP_COUNT)], 1)

    @staticmethod
    def calc(email: Email, weight=1) -> "Vectorizer.Vec":
        content = " ".join(Vectorizer.html_content.extract(email.le_contante))
        capitalised = sum((1 for _ in sf_token.t(content, lambda w: str(w).upper() == str(w))))
        nonsense = sum(1 for _ in sf_token.t(content, lambda w: sf_token.is_nonsense(w)))
        non_ascii = sum(1 for char in content if ord(char) > 255)
        length = len(content)
        links = Vectorizer.link_counter.count(content)
        exclamations = content.count('!')
        headers = len(email.headers)

        return Vectorizer.Vec([headers, capitalised, non_ascii, nonsense, length, links, exclamations], weight)
