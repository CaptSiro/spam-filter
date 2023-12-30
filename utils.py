import math
import random


def read_classification_from_file(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return {line[0]: line[1] for line in (
            line.strip().split() for line in file.readlines()
        ) if len(line) == 2}


def array_shuffle(array):
    length = len(array)
    for i in range(length):
        swap = random.randint(0, length - 1)
        array[i], array[swap] = array[swap], array[i]


def normal_dist(avg, stddev, x):
    if avg == 0 or stddev == 0:
        return 1

    a = (x - avg) ** 2
    b = 2 * (stddev ** 2)
    exp = -(a / b)
    c = (2 * math.pi * stddev) ** 0.5
    return math.e ** exp / c


def r(x):
    return round(x * 100) / 100


if __name__ == "__main__":
    print(read_classification_from_file("test.txt"))
