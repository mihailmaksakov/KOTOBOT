import datetime
import os

import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve

from joblib import dump, load

import pandas as pd

import configparser
import logging

from files import create_file_if_not_exist as check_f

BASE_DIRECTORY = 'userdetector'
CFG_FILE = 'userdetector.ini'
LOG_FILE = 'userdetector.log'
DATETIME_FORMAT = '%Y.%m.%d %H:%M:%S'

HOUR = 60*60

check_f(f'{BASE_DIRECTORY}/{LOG_FILE}')
logging.basicConfig(filename=f'{BASE_DIRECTORY}/{LOG_FILE}', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class UserDetector:

    def __init__(self):
        self.cf = UDConfiguration()
        self.model_file = f'{BASE_DIRECTORY}/{self.cf.model_file_name}'
        self.vectorizer_file = f'{BASE_DIRECTORY}/{self.cf.word_vectorizer_file_name}'
        self.feature_scaler_file = f'{BASE_DIRECTORY}/{self.cf.feature_scaler_file_name}'

    def update_model(self, sql_communicator, verbose=0):

        # get all messages from sql
        precessed_messages = sql_communicator.get_all_processed_data()

        wv = self.init_word_vectorizer(precessed_messages['message_text'])

        X = self.transform_messages(precessed_messages, wv)

        y_tmp = precessed_messages['user_id'].to_numpy()
        y = list()
        all_users = sql_communicator.get_users()
        for y_i in y_tmp:
            u_t_id = all_users.loc[all_users['user_id'] == y_i]['id'].values
            if len(u_t_id) > 0:
                y.append(u_t_id[0])
            else:
                y.append(-1)

        X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.2)

        param_grid = {'C': [0.1, 0.8, 1, 2, 3, 5, 10], 'kernel': ('rbf', )}
        # param_grid = {'C': [10], 'kernel': ('rbf',)}

        estimator = GridSearchCV(
            SVC(), param_grid,
            cv=5, n_jobs=4,
            verbose=verbose
        )

        grid_search_res = estimator.fit(X_train, y_train)
        best_estimator = grid_search_res.best_estimator_

        clf = best_estimator.fit(X_train, y_train)
        score = clf.score(X_test, np.ravel(y_test))

        dump(clf, self.model_file)

        logger.info(f'best estimator search: {grid_search_res.best_params_}')

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=best_estimator,
            X=X,
            y=y, cv=5,
            scoring='neg_mean_squared_error', n_jobs=4, shuffle=True, verbose=verbose)

        train_scores_mean = -train_scores.mean(axis=1)
        validation_scores_mean = -validation_scores.mean(axis=1)

        import matplotlib.pyplot as plt

        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, validation_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title(f'Current model (params: {grid_search_res.best_params_}, score: {score:.3f})', fontsize=18)
        plt.legend()
        plt.ylim(0, 5)

        plt.savefig(f'{BASE_DIRECTORY}/lc_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png')

        return

    @property
    def word_vectorizer(self):
        if os.path.exists(self.vectorizer_file):
            return load(self.vectorizer_file)

    def init_word_vectorizer(self, message_texts):

        from message_processing import normalize_string

        uniq_words = set(
            ' '.join(list(normalize_string(w, None, False)[0] for w in message_texts)).split())

        bad_words = list(w for w in uniq_words if (len(set(w)) < 5 and len(w) > 8) or 'ref_' in w or '_number_' in w)
        words = list(w for w in uniq_words if not w in bad_words and len(set(w)) > 1)

        # message_text_correct = list(w for w in ' '.join(precessed_messages['message_text_processed']).split() if w in words)
        message_texts_correct = list(w for w in ' '.join(message_texts).split() if w in words)

        df_counts = pd.DataFrame(pd.Series(message_texts_correct).value_counts())
        usable_words = df_counts.loc[(df_counts[0] >= 1)]
        vectorizer = CountVectorizer(binary=True)
        vectorizer.fit(usable_words.index.values)

        dump(vectorizer, self.vectorizer_file)

        return vectorizer

    @property
    def feature_scaler(self):
        if os.path.exists(self.feature_scaler_file):
            return load(self.feature_scaler_file)

    def init_feature_scaler(self, features):

        scaler = preprocessing.StandardScaler()
        scaler.fit(features)

        dump(scaler, self.feature_scaler_file)

        return scaler

    def transform_messages(self, messages, wv, fs=None):

        X = wv.transform(messages['message_text']).toarray()
        X_additional = messages[['obscene_terms_count', 'sentiment', 'contains_text',
                                           'contains_sticker', 'contains_image', 'typos_count']].copy().to_numpy()

        X_word_stats = np.array(
            list([len(t.replace(' ', '')) / len(t.split()) if len(t.split()) > 0 else 0, len(t.split())]
                 for t in messages['message_text']))

        X_date_vars = np.array(list([d.hour, d.weekday()] for d in messages['date_time']))

        X = np.hstack(
            (X, X_additional, X_word_stats, X_date_vars))

        if not fs:
            fs = self.init_feature_scaler(X)

        return fs.transform(X)

    def predict_user(self, text):
        # read word vectorizer from file
        # process message
        # read model from file or update model
        # predict user
        pass


class UDConfiguration:

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_file = f'{BASE_DIRECTORY}/{CFG_FILE}'
        self.init_config()

    def init_config(self):

        if not os.path.exists(self.config_file):
            self.create_config()
        else:
            self.read_config()

    def create_config(self):
        self.config.add_section("Settings")
        self.config.set("Settings", "model_update_period", "1")  # hours
        self.config.set("Settings", "model_file_name", "userdetector.clf")
        self.config.set("Settings", "word_vectorizer_file_name", "userdetector.vct")
        self.config.set("Settings", "feature_scaler_file_name", "userdetector.scl")
        # config.set("Settings", "font_info",
        #            "You are using %(font)s at %(font_size)s pt")

        self.config.add_section('Variables')
        self.config.set("Variables", "last_update", "")  # datetime

        self.save_config()

    def read_config(self):
        self.config.read(self.config_file)

    def save_config(self):
        check_f(self.config_file)
        with open(self.config_file, 'w') as config_file:
            self.config.write(config_file)

    def get_setting(self, name):
        return self.config['Settings'][name]

    def get_variable(self, name):
        return self.config['Variables'][name]

    def update_variable(self, name, value):
        self.config.set('Variables', name, value)
        self.save_config()

    @property
    def model_update_period(self):
        """
        gets model update period in seconds
        :return:
        """
        up = self.get_setting('model_update_period')
        if up:
            return int(up)*HOUR
        else:
            return 24*HOUR

    @property
    def model_file_name(self):
        return self.get_setting('model_file_name')

    @property
    def word_vectorizer_file_name(self):
        return self.get_setting('word_vectorizer_file_name')

    @property
    def feature_scaler_file_name(self):
        return self.get_setting('feature_scaler_file_name')

    @property
    def last_update(self):
        last_update_s = self.get_variable('last_update')
        if last_update_s:
            return datetime.strptime(last_update_s, DATETIME_FORMAT)
        else:
            return datetime.time()

    @last_update.setter
    def last_update(self, value):
        date_time = value.strftime(DATETIME_FORMAT)
        self.update_variable('last_update', date_time)


# def test_NN(alphas, hls, activations, X_train, y_train, X_test, y_test):
#
#     import winsound
#
#     for a in alphas:
#         for hl in hls:
#             for act in activations:
#                 best_estimator = MLPClassifier(alpha=a, hidden_layer_sizes=hl, max_iter=1000, activation=act)
#                 clf = best_estimator.fit(X_train, y_train)
#                 print(f'alpha={a}, hidlayers={hl}, activ={act}: {clf.score(X_test, np.ravel(y_test))}')
#                 winsound.Beep(2500, 1000)


if __name__ == '__main__':

    ud = UserDetector()

    from sqlcommunicator import SQLCommunicator
    ud.update_model(SQLCommunicator('localhost', 'kotoboto', 'root'), 2)

    # from sqlcommunicator import SQLCommunicator
    # from message_processing import normalize_string
    #
    # sql_communicator = SQLCommunicator('localhost', 'kotoboto', 'root')
    #
    # precessed_messages = sql_communicator.get_all_processed_data()
    # # process messages
    # # words_uniq = set(' '.join(precessed_messages['message_text_processed']).split())
    # words_uniq = set(' '.join(list(normalize_string(w, None, False)[0] for w in precessed_messages['message_text'])).split())
    #
    # bad_words = list(w for w in words_uniq if (len(set(w)) < 5 and len(w) > 8) or 'ref_' in w or '_number_' in w)
    # words = list(w for w in words_uniq if not w in bad_words and len(set(w)) > 1)
    #
    # # message_text_correct = list(w for w in ' '.join(precessed_messages['message_text_processed']).split() if w in words)
    # message_text_correct = list(w for w in ' '.join(precessed_messages['message_text']).split() if w in words)
    #
    # df_counts = pd.DataFrame(pd.Series(message_text_correct).value_counts())
    # usable_words = df_counts.loc[(df_counts[0] >= 1)]
    # # usable_words = df_counts.loc[(df_counts[0] >= 10) & (df_counts[0] <= 40)]
    # # usable_words = df_counts.loc[(df_counts[0] >= 3)]
    #
    # vectorizer = CountVectorizer(binary=True)
    # vectorizer = vectorizer.fit(usable_words.index.values)
    # X = vectorizer.transform(precessed_messages['message_text']).toarray()
    # # X = vectorizer.transform(precessed_messages['message_text_processed']).toarray()
    #
    # # pickle.dump(vocabulary, open("spam_assasin_based_spam_filter/vocabulary.pkl", "wb"))
    #
    # X_additional = precessed_messages[['obscene_terms_count', 'sentiment', 'contains_text',
    #                                    'contains_sticker', 'contains_image', 'typos_count']].copy().to_numpy()
    #
    # X_word_stats = np.array(list([len(t.replace(' ', '')) / len(t.split()) if len(t.split()) > 0 else 0, len(t.split())]
    #                         for t in precessed_messages['message_text_processed']))
    #
    # X_date_vars = np.array(list([d.hour, d.weekday()] for d in precessed_messages['date_time']))
    #
    # X = np.hstack(
    #     (X, X_additional, X_word_stats, X_date_vars))
    #
    # scaler = preprocessing.StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    #
    # y = precessed_messages['user_id'].to_numpy()
    #
    # y_id = list()
    # all_users = sql_communicator.get_users()
    # for y_i in y:
    #     u_t_id = all_users.loc[all_users['user_id']==y_i]['id'].values
    #     if len(u_t_id)>0:
    #         y_id.append(u_t_id[0])
    #     else:
    #         y_id.append(-1)
    # # y_id = list([0] for y_i in y)
    #
    # from sklearn.model_selection import learning_curve
    #
    # best_estimator = SVC(kernel='rbf', C=2)
    #
    # train_sizes, train_scores, validation_scores = learning_curve(
    #     estimator=best_estimator,
    #     X=X,
    #     y=y_id, cv=3,
    #     scoring='neg_mean_squared_error', n_jobs=8, shuffle=True, verbose=2)
    #
    # train_scores_mean = -train_scores.mean(axis=1)
    # validation_scores_mean = -validation_scores.mean(axis=1)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, np.array(y_id), test_size=0.2)
    #
    # clf = best_estimator.fit(X_train, y_train)
    # print(f'score: {clf.score(X_test, np.ravel(y_test))}')
    #
    # import matplotlib.pyplot as plt
    #
    # plt.style.use('seaborn')
    # plt.plot(train_sizes, train_scores_mean, label='Training error')
    # plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    # plt.ylabel('MSE', fontsize=14)
    # plt.xlabel('Training set size', fontsize=14)
    # # plt.title('Learning curves for a linear regression model', fontsize=18, y=1.03)
    # plt.legend()
    # plt.ylim(0, 5)
    #
    # plt.show()
    #
    # exit(1)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_minmax, np.array(y_id), test_size=0.2)
    #
    # # param_grid = {'C': [0.01, 0.1, 0.8, 1, 10, 100, 1000],
    # #               'gamma': [0.01, 0.1, 1, 10], }
    # # param_grid = {'C': [0.01, 0.1, 0.8, 1, 10],
    # #               'gamma': [0.01, 0.1, 1, 10], }
    # # param_grid = {'C': [0.01], }
    # # param_grid = {'C': [0.8, 1, 2, 3, 5, 10, 100, 1000, 10000], 'kernel': ('rbf', )}
    # #
    # # estimator = GridSearchCV(
    # #     SVC(), param_grid,
    # #     cv=3, n_jobs=4
    # # )
    # #
    # # grid_search_res = estimator.fit(X_train, y_train)
    # # best_estimator = grid_search_res.best_estimator_
    # #
    # # print(f'best params: {grid_search_res.best_params_}')
    #
    # # test_NN([0.001, 0.01, 0.1, 1, 10, 100], [(10,), (20,), (50,), (100,), (50,20, )], ['relu', ], X_train, y_train, X_test, y_test)
    #
    # best_estimator = SVC(kernel='rbf', C=10)
    # from sklearn import metrics
    #
    # # clf = best_estimator.fit(X_train, y_train)
    # # print(clf.score(X_test, np.ravel(y_test)))
    # #
    # # import winsound
    # # winsound.Beep(2500, 1000)
    # # exit(0)
    #
    # # dump(best_estimator, 'spam_assasin_based_spam_filter/spam_assasin.linear.clf')
    # # verify model
    # train_errors = np.empty((0, 2))
    # validation_errors = np.empty((0, 2))
    #
    # import math
    #
    # for m in range(10, X_train.shape[0], int(math.pow(10, math.log10(X_train.shape[0]) - 1))):
    #     current_mse_train = np.empty(2)
    #     current_mse_test = np.empty(2)
    #     for j in range(2):
    #         indices = np.random.choice(X_train.shape[0], size=m + 1, replace=False)
    #         X_tmp = X_train[indices, :]
    #         y_tmp = y_train[indices]
    #         clf = best_estimator.fit(X_tmp, y_tmp)
    #         current_mse_train[j - 1] = mean_squared_error(clf.predict(X_tmp), y_tmp)
    #         x_test_pred = clf.predict(X_test)
    #         current_mse_test[j - 1] = mean_squared_error(x_test_pred, y_test)
    #         print(metrics.accuracy_score(y_test, x_test_pred), f'{m} of {X_train.shape[0]}')
    #
    #     train_errors = np.vstack((train_errors, np.array([[np.mean(current_mse_train), m]])))
    #     validation_errors = np.vstack((validation_errors, np.array([[np.mean(current_mse_test), m]])))
    #
    #     print(f'min temp test error: {min(validation_errors[:, 0])}')
    #
    # print(f'min test error: {min(validation_errors[:, 0])}')
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    #
    # plt.plot(train_errors[:, 1], train_errors[:, 0], label='train')
    # plt.plot(validation_errors[:, 1], validation_errors[:, 0], label='test')
    # leg = plt.legend()
    #
    # plt.show()
