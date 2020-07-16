import datetime
import os

import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer

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

    def update_model(self, sql_communicator):
        # get all messages from sql
        return
        precessed_messages = sql_communicator.get_all_processed_texts()
        # process messages
        # all_words = ' '.join(precessed_messages['message_text_processed'])

        vectorizer = CountVectorizer(binary=True)
        vectorizer.fit_transform(precessed_messages['message_text_processed'])

        # pickle.dump(vocabulary, open("spam_assasin_based_spam_filter/vocabulary.pkl", "wb"))

        X = vectorizer.transform(precessed_messages['message_text_processed']).toarray()
        y = precessed_messages['user_id'].to_numpy()

        min_max_scaler = preprocessing.MinMaxScaler()
        X_minmax = min_max_scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2)

        param_grid = {'C': [0.01, 0.1, 0.8, 1, 10],
                      'gamma': [0.01, 0.1, 1, 10], }

        estimator = GridSearchCV(
            SVC(kernel='linear'), param_grid
        )

        grid_search_res = estimator.fit(X_train, y_train)
        best_estimator = grid_search_res.best_estimator_

        # dump(best_estimator, 'spam_assasin_based_spam_filter/spam_assasin.linear.clf')
        # verify model
        train_errors = np.empty((0, 2))
        validation_errors = np.empty((0, 2))

        for m in range(10, X_train.shape[0], 10):
            current_mse_train = np.empty(2)
            current_mse_test = np.empty(2)
            for j in range(2):
                indices = np.random.choice(X_train.shape[0], size=m + 1, replace=False)
                X_tmp = X_train[indices, :]
                y_tmp = y_train[indices]
                clf = best_estimator.fit(X_tmp, y_tmp)
                current_mse_train[j - 1] = mean_squared_error(clf.predict(X_tmp), y_tmp)
                current_mse_test[j - 1] = mean_squared_error(clf.predict(X_test), y_test)

            train_errors = np.vstack((train_errors, np.array([[np.mean(current_mse_train), m]])))
            validation_errors = np.vstack((validation_errors, np.array([[np.mean(current_mse_test), m]])))

        print(f'min test error: {min(validation_errors[:, 0])}')

        import matplotlib.pyplot as plt

        plt.figure()

        plt.plot(train_errors[:, 1], train_errors[:, 0], label='train')
        plt.plot(validation_errors[:, 1], validation_errors[:, 0], label='test')
        leg = plt.legend()

        plt.show()

        # asd = vectorizer.transform([precessed_messages.iloc[0, :]['message_text_processed']]).toarray()

        # words = pd.Series(all_words.split()).value_counts()
        # learn model
        # print(self.model_file)
        # write model to file

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
        self.config.set("Settings", "model_file_name", "model.clf")  # hours
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

