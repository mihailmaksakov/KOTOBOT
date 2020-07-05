import re

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

from pyaspeller import Word

from obscene_words_filter.conf import bad_words, good_words
from obscene_words_filter.regexp import build_good_phrase, build_bad_phrase
from obscene_words_filter.words_filter import ObsceneWordsFilter

from bs4 import BeautifulSoup

import numpy as np

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

TELEGRAM_USER_REF_RE = r'<a href="tg://user[\?]id=([\d]+)">[\w]+</a>'


class UserDetector:

    def __init__(self, model_dir='usersimilarity'):
        self.model_file = model_dir + '/model.clf'

    def update_model(self, sql_communicator):
        # get all messages from sql
        precessed_texts = sql_communicator.get_all_processed_texts()
        # process messages

        # learn model
        print(self.model_file)
        # write model to file
        pass

    # def init_clf(self):
    #     X = np.empty((0,0))
    #     y = np.empty((0))
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #     return 1

    def predict_user(self, text):
        # process message
        # read model from file or update model
        # predict user
        pass


def raw_message_to_string(message_text):
    if message_text:
        return str(message_text)
    else:
        return ''


def normalize_string(text):

    soup = BeautifulSoup(text.lower(), features="html.parser")

    __string = soup.get_text()
    __string = clean_typos(__string)
    __string = re.sub(r'http[s]?://[\w\-_.=\&\?/]+', ' _httpref_ ', __string)
    __string = re.sub(r'[\w\-_]+@[\w\-_.]+', ' _emailref_ ', __string)
    __string = re.sub(r'[^а-яА-Яa-zA-Z0-9_\'\-’]+', ' ', __string)
    __string = re.sub(r'\s\d+\s', ' _number_ ', __string)
    __string = re.sub(TELEGRAM_USER_REF_RE, ' _userref_ ', __string)
    # <a href= "tg://user?id=829112612">Roman</a>

    # parse_result = ' '.join([lemmatizer.lemmatize(word, tag_dict.get(tag[0], wordnet.NOUN)) for word, tag in pos_tag(__string.split())])

    return __string


def clean_typos(text):
    result = []
    for w in text.split():
        check = Word(w)
        if check.correct \
                or len(check.variants) == 0:
            result.append(w)
        else:
            result.append(check.variants[0])

    return ' '.join(result)


def content_is_empty(content):
    if content:
        return True
    else:
        return False


def get_obscene_terms(text):

    bad_words.append(build_bad_phrase('о х у е н'))

    bad_words_re = re.compile('|'.join(bad_words), re.IGNORECASE | re.UNICODE)
    good_words_re = re.compile('|'.join(good_words), re.IGNORECASE | re.UNICODE)

    bad_filter = ObsceneWordsFilter(bad_words_re, good_words_re)

    return list(b.group() for b in bad_filter.find_bad_word_matches(text))


def get_obscene_terms_count(text):
    return len(get_obscene_terms(text))


def get_sentiment(text):

    if not text:
        return 5

    model = FastTextSocialNetworkModel(tokenizer=RegexTokenizer(), lemmatize=True)

    results = model.predict([text], k=2)
    result = 0
    # result = result + 0.5*(results[0]['neutral'] if 'neutral' in results[0] else 0)
    result = result + (results[0]['positive'] if 'positive' in results[0] else 0)
    result = result - (results[0]['negative'] if 'negative' in results[0] else 0)

    return (result + 1) * 10


def other_user_similarity(text):
    return -1


def get_message_topic(text, sql_communicator):
    return ''


def get_message_recipient(text, reply_to_message_id, sql_communicator):

    result = 0

    if reply_to_message_id:
        result = sql_communicator.get_user_id_by_message_id(reply_to_message_id)
    if not result:
        re_result = re.match(TELEGRAM_USER_REF_RE, text)
        if re_result:
            if re_result.group(1):
                return int(re_result.group(1))

    return result
