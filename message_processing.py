import re

import numpy as np

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

from yaspellerextension import YandexSpellerExt

from obscene_words_filter.conf import bad_words, good_words
from obscene_words_filter.regexp import build_good_phrase, build_bad_phrase
from obscene_words_filter.words_filter import ObsceneWordsFilter

import pymorphy2

from bs4 import BeautifulSoup

from userdetector import UserDetector as ud

TELEGRAM_USER_REF_RE = r'<a href="tg://user[\?]id=([\d]+)">[\w]+</a>'


def raw_message_to_string(message_text):
    if message_text:
        return str(message_text)
    else:
        return ''


def normalize_string(text, sql_c, lemmatize=True):

    __string = text
    __string = re.sub(r'http[s]?://[\w\-_.=\&\?/]+', ' _httpref_ ', __string)
    __string = re.sub(r'[\w\-_]+@[\w\-_.]+.[\w\-_]+', ' _emailref_ ', __string)
    __string = re.sub(r'[^а-яёЁА-Яa-zA-Z0-9_\'\-’]+', ' ', __string)
    __string = re.sub(TELEGRAM_USER_REF_RE, ' _userref_ ', __string)
    # <a href= "tg://user?id=829112612">Roman</a>

    typos_count = 0
    if lemmatize:
        clean_text = clean_typos(__string, sql_c)
        typos_count = len(list(w for w in __string.lower().split() if not w in clean_text.lower()))
        __string = clean_text
        __string = re.sub(r'[^а-яёЁА-Яa-zA-Z0-9_\'\-’]+', ' ', __string)
        __string = re.sub(r'\s?\d+\s', ' _number_ ', __string)
        __string = ' '.join([lemmatize_word(w) for w in __string.split()])

    return __string, typos_count


def lemmatize_word(word):
    p = pymorphy2.MorphAnalyzer().parse(word)
    return max(p, key=lambda x: x.score).normal_form if p else word


def clean_typos(text, sql_c):
    return YandexSpellerExt(sql_c).clean_typos(text)


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
    result = result + (min(results[0]['positive'], 1) if 'positive' in results[0] else 0)
    result = result - (min(results[0]['negative'], 1) if 'negative' in results[0] else 0)

    return round((result + 1) * 5)


def other_user_similarity(text):
    user_detector = ud()
    return 0


def get_message_topic(text, sql_communicator):
    return ''


def get_message_recipient(text, reply_to_message_id, sql_c):

    result = 0

    if not reply_to_message_id:
        return result

    if not np.isnan(reply_to_message_id):
        result = sql_c.get_user_id_by_message_id(reply_to_message_id)
    if not result:
        re_result = re.match(TELEGRAM_USER_REF_RE, text)
        if re_result:
            if re_result.group(1):
                return int(re_result.group(1))

    return result
