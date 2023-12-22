import confmat
import utils


def quality_score(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + 10 * fp + fn)

def compute_quality_for_corpus(corpus_dir):
    truth_dict = utils.read_classification_from_file(f"{corpus_dir}/!truth.txt")
    prediction_dict = utils.read_classification_from_file(f"{corpus_dir}/!prediction.txt")

    matrix = confmat.BinaryConfusionMatrix(pos_tag="SPAM", neg_tag="OK")
    matrix.compute_from_dicts(truth_dict, prediction_dict)

    return quality_score(**matrix.as_dict())



if __name__ == "__main__":
    print(compute_quality_for_corpus("1"))
