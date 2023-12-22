def read_classification_from_file(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return {line[0]: line[1] for line in (
            line.strip().split() for line in file.readlines()
        ) if len(line) == 2}


if __name__ == "__main__":
    print(read_classification_from_file("test.txt"))
