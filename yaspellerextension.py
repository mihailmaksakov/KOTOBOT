from pyaspeller import Word


class YandexSpellerExt:

    def __init__(self, sql_communicator):
        self.sqlc = sql_communicator
        # self.dict_cache = self.sqlc.

    def check_dict(self, word):
        return self.sqlc.get_word_from_typo_d(word=word)

    def update_dict(self, word, correct_word, check=0):
        return self.sqlc.put_word_to_typo_d(word=word, correct_word=correct_word, check=check)

    def clean_typos(self, text):

        result = []

        for w in text.split():
            dict_w = self.check_dict(w)
            if dict_w:
                result.append(dict_w)
            else:
                check = Word(w)
                if check.correct \
                        or len(check.variants) == 0:
                    correct_word = w
                else:
                    correct_word = check.variants[0]
                result.append(correct_word)
                self.update_dict(w, correct_word)

        return ' '.join(result)


if __name__ == '__main__':

    from sqlcommunicator import SQLCommunicator
    sql_c = SQLCommunicator('localhost', 'kotoboto', 'root')

    ys = YandexSpellerExt(sql_c)

    from special_words_import import special_words
    for k, v in special_words.items():
        ys.update_dict(k, v, 1)