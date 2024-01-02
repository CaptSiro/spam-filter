import re
from html.parser import HTMLParser



__STOPWORDS__ = None

def get_stopwords():
    global __STOPWORDS__
    stopwords = __STOPWORDS__

    if stopwords is not None:
        return stopwords

    stopwords = set()
    with open("stopwords.txt") as file:
        for word in file.readlines():
            stopwords.add(word.strip())

    __STOPWORDS__ = stopwords
    return __STOPWORDS__


def tokenize_word(content: str, index, length):
    word = ""

    while True:
        if not index < length:
            break

        if not content[index].isalpha() and content[index] != '-':
            break

        word += content[index]
        index += 1

    return word, index


def tokenize(text_content: str):
    content = text_content.lower()
    tokens = []
    i = 0
    length = len(content)
    stopwords = get_stopwords()

    while i < length:
        if content[i].isalpha():
            word, index = tokenize_word(content, i, length)
            i = index

            if word not in stopwords:
                tokens.append(word)

            continue

        i += 1

    return tokens


def t(content: str, predicate):
    length = len(content)
    i = 0

    while i < length:
        if content[i].isalpha():
            word, index = tokenize_word(content, i, length)
            i = index

            if predicate(word):
                yield word

            continue

        i += 1


class HTMLTokenStream(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_contents = []
        self.stopwords = get_stopwords()
        self.token_predicate = lambda w: w not in self.stopwords

    def handle_data(self, data):
        self.text_contents.append(data)

    def stream(self, data):
        self.text_contents = []
        self.feed(data)

        return (s for text_content in self.text_contents for s in t(text_content.lower(), self.token_predicate))


class HTMLLinkCounter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = 0
        self.url_re = re.compile(r"(https?://(?:www\.|(?!www))(?:[a-zA-Z0-9]+:[a-zA-Z0-9]+@)?[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.\S]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.\S{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.\S{2,}|www\.[a-zA-Z0-9]+\.\S{2,})")

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        self.links += 1

    def handle_data(self, data):
        self.links += len(self.url_re.findall(data))

    def count(self, data) -> int:
        self.links = 0
        self.feed(data)
        return self.links


class HTMLContent(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_contents = []

    def handle_data(self, data):
        self.text_contents.append(data)

    def extract(self, data) -> [str]:
        self.text_contents = []
        self.feed(data)
        return self.text_contents
