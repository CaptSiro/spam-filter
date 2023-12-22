class BinaryConfusionMatrix:
    def __init__(self, pos_tag, neg_tag):
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag
        self.matrix = [[0, 0], [0, 0]]

    def hash(self, tag):
        if tag == self.pos_tag:
            return 0

        if tag == self.neg_tag:
            return 1

        raise ValueError(f"Must be either '{self.pos_tag}' of '{self.neg_tag}'")

    def as_dict(self):
        is_spam = self.hash(self.pos_tag)
        is_not_spam = self.hash(self.neg_tag)

        return {
            'tp': self.matrix[is_spam][is_spam],
            'tn': self.matrix[is_not_spam][is_not_spam],
            'fp': self.matrix[is_not_spam][is_spam],
            'fn': self.matrix[is_spam][is_not_spam],
        }

    def update(self, truth, prediction):
        i = self.hash(truth)
        j = self.hash(prediction)

        self.matrix[i][j] = self.matrix[i][j] + 1

    def compute_from_dicts(self, truth_dict, prediction_dict):
        with open("./true.txt", "w", encoding="utf8") as tp:
            for key, truth in truth_dict.items():
                if truth != prediction_dict[key]:
                    tp.write(f"{key} {truth} {prediction_dict[key]}\n")
                self.update(truth, prediction_dict[key])
