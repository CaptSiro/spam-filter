import math


class WordCounter:
    def __init__(self, tokenizer):
        self.terms = {}
        self.documents = []
        self.matrix = {}
        self.tokenizer = tokenizer

    def scan(self, document: str, is_ok=True) -> dict[str, int]:
        frequencies = {}

        for token in self.tokenizer(document):
            if frequencies.get(token) is None:
                frequencies[token] = 1 if is_ok else -1
            else:
                frequencies[token] += 1 if is_ok else -1

        for term in frequencies:
            if term in self.terms:
                self.terms[term] += 1
            else:
                self.terms[term] = 1

        return frequencies

    def vec(self, document: str) -> dict[str, int]:
        frequencies = {}
        count = 0

        for token in self.tokenizer(document):
            count += 1
            if frequencies.get(token) is None:
                frequencies[token] = 1
            else:
                frequencies[token] += 1

        return {term: frequencies[term] / count for term in frequencies}

    def idf(self):
        """Inverse document frequency"""
        return {term: math.log10(len(self.documents) / self.terms[term]) for term in self.terms}

    def tfidf(self):
        """Term frequency * Inverse document frequency"""
        idf = self.idf()
        return [{term: doc[term] * idf[term] for term in doc} for doc in self.documents]

    def tfidf_vec(self):
        """Term frequency * Inverse document frequency as single vector"""
        idf = self.idf()
        return {term: sum([doc.get(term, 0) * idf[term] for doc in self.documents]) / len(self.documents) for term in self.terms}




if __name__ == "__main__":
    from tokenize import HTMLTokenStream
    def main():
        tokenizer = HTMLTokenStream()
        counter = WordCounter(tokenizer.stream)
        counter.documents.append(counter.scan("i love natural language processing but i hate python"))
        counter.documents.append(counter.scan("i like image processing", False))
        counter.documents.append(counter.scan("i like signal processing and image processing"))

        max_len = max([len(token) for token in counter.terms])
        padding = f"%{max_len + 1}s"

        for term in counter.terms:
            print(padding % term, end='')

        print()
        for doc in counter.documents:
            print()
            for term in counter.terms:
                print(padding % doc.get(term, 0), end='')

        print()

        print()
        idf = counter.idf()
        for term in idf:
            print(padding % ("%.5f" % idf[term]), end='')

        print()

        # calculate idf vector and multiply it with documents
        for doc in counter.tfidf():
            print()
            for term in counter.terms:
                print(padding % ("%.5f" % doc.get(term, 0)), end='')

        print()

        # sum(tfidf_matrix)
        print()
        vec = counter.tfidf_vec()
        for term in vec:
            print(padding % ("%.5f" % vec[term]), end='')

    main()