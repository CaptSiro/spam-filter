import math
import filter
import quality


def is_ok(spec, file):
    if file not in spec:
        print(f"{file} not in spec")
        return True

    return spec[file] == "OK"

def scal_mul(vec_x: dict, vec_y: dict):
    return sum((vec_x.get(term, 0) * vec_y.get(term, 0) for term in vec_x))

def main():
    # corpus.Corpus.random_corpus("./corpus")
    train_dir = "./1/"
    # train_dir = "./corpus/train/"
    test_dir = "./2/"
    # test_dir = "./corpus/test/"

    print(f"Training on directory: {train_dir}")
    print(f"Testing on directory: {test_dir}")

    f = filter.MyFilter()
    f.train(train_dir)
    f.test(test_dir)

    print(f"{f.__class__.__name__} {math.floor(quality.compute_quality_for_corpus(test_dir) * 100 * 100) / 100}%")

    f = filter.MyFilterPropAnalysis()
    f.train(train_dir)
    f.test(test_dir)

    print(f"{f.__class__.__name__} {math.floor(quality.compute_quality_for_corpus(test_dir) * 100 * 100) / 100}%")

    # f = filter.MyFilterPropAnalysisOnly()
    # f.train(train_dir)
    # f.test(test_dir)
    #
    # print(f"{f.__class__.__name__} {math.floor(quality.compute_quality_for_corpus(test_dir) * 100 * 100) / 100}%")



if __name__ == '__main__':
    main()
