import corpus
import filter
import quality
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
    train_dir = "1"
    test_dir = "2"

    f = filter.MyFilter()
    f.train(train_dir)
    f.test(test_dir)

    print(quality.compute_quality_for_corpus(test_dir))


if __name__ == '__main__':
    main()
