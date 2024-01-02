class Email:
    def __init__(self, le_contante, headers):
        self.le_contante = le_contante
        self.headers = headers

    @staticmethod
    def from_file(file):
        previous_key = ""
        headers = {}
        while True:
            line = file.readline()
            if line == '\n':
                break
            if line[0].isspace():
                headers[previous_key] += line.strip()
                continue
            parts = line.split(': ', 1)
            headers[parts[0]] = parts[1].strip()
            previous_key = parts[0]
        return Email(file.read().strip(), headers)


if __name__ == "__main__":
    with open("1", 'r', encoding='utf-8') as f:
        xd = Email.from_file(f)
        print("headers: ", xd.headers, "\ncontante:\n", xd.le_contante)
