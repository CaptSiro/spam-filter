import math
import time
import sf_corpus
import filter
import quality


def is_ok(spec, file):
    if file not in spec:
        print(f"{file} not in spec")
        return True

    return spec[file] == "OK"

def scal_mul(vec_x: dict, vec_y: dict):
    return sum((vec_x.get(term, 0) * vec_y.get(term, 0) for term in vec_x))

def time_fn(fn, *args, **kwargs):
    start = time.time()
    fn(*args, **kwargs)
    print("%-16s: %.3fs" % (fn.__name__, time.time() - start))

def print_stats(f, test_dir: str) -> None:
    percentage, matrix = quality.compute_quality_for_corpus(test_dir)
    print(f"{f.__class__.__name__} {math.floor(percentage * 100 * 100) / 100}%")
    print(matrix)

def main():
    # sf_corpus.Corpus.random_corpus("./corpus")
    train_dir = "./1/"
    # train_dir = "./corpus/train/"
    test_dir = "./2/"
    # test_dir = "./corpus/test/"

    print(f"Training on directory: {train_dir}")
    print(f"Testing on directory: {test_dir}")

    f = filter.MyFilter()
    time_fn(f.train, train_dir)
    f.test(test_dir)

    print_stats(f, test_dir)



if __name__ == '__main__':
    main()
