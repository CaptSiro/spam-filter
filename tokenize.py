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

    while True:
        if length <= i:
            break

        if content[i].isalpha():
            word, index = tokenize_word(content, i, length)
            i = index

            if word not in stopwords:
                tokens.append(word)

            continue

        i += 1



    return tokens



class HTMLTokenStream(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_contents = []

    def handle_data(self, data):
        self.text_contents.append(data)

    def stream(self, data):
        self.text_contents = []
        self.feed(data)

        return (s for text_content in self.text_contents for s in tokenize(text_content))
