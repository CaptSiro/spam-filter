import re


class SenderCounter:

    def __init__(self):
        # [appearance_count, score]
        self.appearances = {}

    def load_sender(self, headers_dict, is_ok, weight=1):
        sender_address = SenderCounter.find_mail(headers_dict)
        if sender_address not in self.appearances:
            self.appearances[sender_address] = [1, 0]
        else:
            self.appearances[sender_address][0] += 1
        self.appearances[sender_address][1] += weight if is_ok else -weight

    def result(self):
        result = {}
        for key in self.appearances.keys():
            result[key] = self.appearances[key][1]/self.appearances[key][0]
        return result

    def test_sender(self, headers_dict):
        sender_address = SenderCounter.find_mail(headers_dict)
        value = self.appearances.get(sender_address, [1, 0])
        return value[1]/value[0]

    @staticmethod
    def find_mail(headers_dict):
        sender_address = re.search(r'(([\w.?+-]+)|([\w.?+-]*"[\w.?@+-]+"[\w.?+-]*))@(([\w-]+\.[\w.-]+)|(\[[0-9]+]))', headers_dict["from"])

        if sender_address is None:
            return None

        sender_address = sender_address.group()
        return sender_address
