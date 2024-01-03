import math

import sf_token


class UniqueCounter:
    def __init__(self):
        self.map = {}
        self.total = 0
        self.unique = 0

    def scan(self, term, times=1) -> None:
        if term in self.map:
            self.map[term] += times
        else:
            self.map[term] = times
            self.unique += 1

        self.total += times

    def get(self, term, default=0) -> int:
        return self.map.get(term, default)



class WordCounter:
    def __init__(self, tokenize, dataset: set[str]):
        self.words = UniqueCounter()
        self.tokenize = tokenize
        self.scanned = 0
        self.dataset = dataset

    def scan(self, string: str) -> UniqueCounter:
        frequencies = UniqueCounter()

        for token in self.tokenize(string):
            frequencies.scan(token)

        return frequencies

    def __lshift__(self, other: UniqueCounter):
        self.scanned += 1

        for token in other.map:
            self.words.scan(token, other.get(token))
            if token not in self.dataset:
                self.dataset.add(token)

    def probability(self, document: UniqueCounter):
        return sum(math.log10(
            (self.words.get(term) + 1) / (self.words.total + len(self.dataset) + 1)
        ) * document.get(term) for term in document.map)


if __name__ == "__main__":
    tokenizer = sf_token.HTMLTokenStream()
    dataset = set()
    spam = WordCounter(tokenizer.stream, dataset)
    ok = WordCounter(tokenizer.stream, dataset)

    spam << spam.scan("dear friend money")
    spam << spam.scan("dear money")
    spam << spam.scan("money")
    spam << spam.scan("money")

    ok << ok.scan("dear friend lunch money")
    ok << ok.scan("dear friend lunch")
    ok << ok.scan("dear friend lunch")
    ok << ok.scan("dear friend")
    ok << ok.scan("dear friend")
    ok << ok.scan("dear")
    ok << ok.scan("dear")
    ok << ok.scan("dear")

    # print(spam.words.map, len(spam.dataset), spam.words.total)
    # print(ok.words.map, len(spam.dataset), ok.words.total)

    total = spam.scanned + ok.scanned

    # print(f"p(N|dear) = {math.pow(10, ok.probability(dear))}")
    # print(f"p(S|dear) = {math.pow(10, spam.probability(dear))}")
    # print(f"p(N|friend) = {math.pow(10, ok.probability(friend))}")
    # print(f"p(S|friend) = {math.pow(10, spam.probability(friend))}")

    test = spam.scan("lunch money money money money")
    print("%-8.6f" % math.pow(10, math.log10(ok.scanned / total) + ok.probability(test)))
    # print("%-8.5f" % math.pow(10, ok.probability(test)))
    print("%-8.6f" % math.pow(10, math.log10(spam.scanned / total) + spam.probability(test)))
    # print("%-8.5f" % math.pow(10, spam.probability(test)))
