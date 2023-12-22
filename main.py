import corpus
import tfidf
import utils
from email_reader import Email
from tokenize import HTMLTokenStream



def is_ok(spec, file):
    if file not in spec:
        print(f"{file} not in spec")
        return True

    return spec[file] == "OK"

def scal_mul(vec_x: dict, vec_y: dict):
    return sum((vec_x.get(term, 0) * vec_y.get(term, 0) for term in vec_x))

def main():
    train_dir = "./1/"
    spec = utils.read_classification_from_file(f"{train_dir}!truth.txt")
    tokenizer = HTMLTokenStream()
    counter = tfidf.WordCounter(tokenizer.stream)

    for file, content in corpus.Corpus(train_dir).emails():
        with open(train_dir + file, "r", encoding="utf8") as fp:
            email = Email.from_file(fp)
            counter.documents.append(counter.scan(email.le_contante, is_ok(spec, file)))

    filter_vec = counter.tfidf_vec()

    real_dir = "./1/"
    with open("./stats.txt", "w", encoding="utf8") as sp:
        with open(real_dir + "!prediction.txt", "w", encoding="utf8") as wp:
            for file, content in corpus.Corpus(real_dir).emails():
                with open(real_dir + file, "r", encoding="utf8") as fp:
                    email = Email.from_file(fp)
                    email_vec = counter.vec(email.le_contante)

                    result = scal_mul(email_vec, filter_vec)
                    sp.write(f"{file} {result}\n")
                    wp.write(f"{file} {'SPAM' if result < -0.05 else 'OK'}\n")


if __name__ == '__main__':
    main()
