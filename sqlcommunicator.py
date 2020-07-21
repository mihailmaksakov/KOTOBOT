# import mysql.connector

import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

from sqlalchemy import Column, Integer, BINARY, String, ForeignKey, DATETIME, func
from sqlalchemy.dialects import mysql as mysqld

from sqlalchemy.exc import SQLAlchemyError

SQL_ENGINE = 'mysql+pymysql'

import pandas as pd


class User(Base):

    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    name = Column(String)
    sex = Column(mysqld.TINYINT)

    def __repr__(self):
        return f'<User({self.name}, {self.sex_repr()}, {self.user_id})>'

    def sex_repr(self):
        return 'Female' if self.sex else 'Male'


class RawMessage(Base):

    __tablename__ = 'raw_chat_data'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    chat_id = Column(mysqld.BIGINT)
    date_time = Column(DATETIME)
    message_text = Column(mysqld.MEDIUMTEXT)
    message_id = Column(mysqld.BIGINT)
    sticker = Column(String(150))
    has_photo = Column(mysqld.TINYINT)
    photo = Column(mysqld.MEDIUMBLOB)
    reply_to_message_id = Column(mysqld.BIGINT)

    user = relationship("User", backref="messages")

    def __repr__(self):
        return f'<RAW Message({self.user}: {self.message_text}, {self.message_id})>'


class RawMessageProcessingQueue(Base):

    __tablename__ = 'raw_data_processing'

    raw_chat_data_id = Column(Integer, ForeignKey('raw_chat_data.id'), primary_key=True)
    processed_data_id = Column(Integer, ForeignKey('processed_chat_data.id'), nullable=True)

    temp_id = 0

    raw_message = relationship("RawMessage", backref="message_processing_queue")
    processed_message = relationship("ProcessedMessage", backref="message_processing_queue")

    def __repr__(self):
        return f'<Message processing queue({self.raw_chat_data_id}, {self.processed_data_id})>'


class ProcessedMessage(Base):

    __tablename__ = 'processed_chat_data'

    id = Column(Integer, primary_key=True)
    obscene_terms_count = Column(mysqld.TINYINT(unsigned=True))
    sentiment = Column(mysqld.TINYINT(unsigned=True))
    other_user_similarity = Column(Integer, ForeignKey('users.user_id'))
    contains_text = Column(mysqld.TINYINT)
    contains_sticker = Column(mysqld.TINYINT)
    contains_image = Column(mysqld.TINYINT)
    message_recipient = Column(Integer, ForeignKey('users.user_id'))
    recipient_is_bot = Column(mysqld.TINYINT)
    topic = Column(String(100))
    image_topic = Column(String(100))
    image_adult = Column(mysqld.TINYINT)
    message_text = Column(mysqld.MEDIUMTEXT)
    message_text_processed = Column(mysqld.MEDIUMTEXT)
    typos_count = Column(Integer)

    temp_id = 0

    predicted_user = relationship("User", foreign_keys=[other_user_similarity])
    recipient = relationship("User", foreign_keys=[message_recipient])

    def __repr__(self):
        return f'<Processed Message({self.message_text_processed}, {self.id})>'

    def to_dict(self):
        t_d = self.__dict__.copy()
        t_d.pop('_sa_instance_state')
        return t_d


class TypoDictionary(Base):

    __tablename__ = 'typo_dictionary'

    id = Column(Integer, primary_key=True)
    word = Column(String(100), index=True)
    correct_word = Column(String(100))
    check = Column(mysqld.TINYINT)

    def __repr__(self):
        return f'<TypoDictionary({self.word} → {self.correct_word})>'


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class SQLConnectionError(Error):

    def __init__(self, host, database, user, cause):
        super(SQLConnectionError, self).__init__()
        self.host = host
        self.database = database
        self.user = user
        self.cause = cause

    def __str__(self):
        return f'Error connecting to SQL server\nhost={self.host}, database={self.database}, user={self.user}\n{self.cause}'


class SQLOperationError(Error):

    def __init__(self, operation_text, cause):
        super(SQLOperationError, self).__init__()
        self.operation_text = operation_text
        self.cause = cause

    def __str__(self):
        return f'Error executing SQL operation\n{self.operation_text}\n{self.cause}'


class SQLCommunicator:

    def __init__(self, host, database, user, password=''):
        self._engine = create_engine(f'{SQL_ENGINE}://{user}:{password}@{host}/{database}')

    def __enter__(self):
        return self

    def update_users(self, telegram_user):

        session = Session(bind=self._engine)

        res = session.query(User).filter(User.user_id == telegram_user.id).first()

        if not res:
            new_user = User(user_id=telegram_user.id, name=telegram_user.name, sex=-1)
            session.add(new_user)
            session.commit()

        session.close()

    def put_message(self, user_id, chat_id, date_time, message_id, message_text, reply_to_message_id, sticker,
                    has_photo=0, photo_binary=None):

        session = Session(bind=self._engine)

        new_message = RawMessage(user_id=user_id, chat_id=chat_id, date_time=date_time,
                                 message_text=message_text, message_id=message_id,
                                 sticker=sticker, has_photo=has_photo, photo=photo_binary,
                                 reply_to_message_id=reply_to_message_id)

        session.add(new_message)
        session.flush()

        self.put_message_to_processing(session, new_message.id)

        session.commit()
        session.close()

    def put_message_to_processing(self, session, message_id):

        do_comm = not session
        if not session:
            session = Session(bind=self._engine)

        new_queue_message = RawMessageProcessingQueue(raw_chat_data_id=message_id)
        session.add(new_queue_message)

        if do_comm:
            session.commit()
            session.close()

    def prepare_raw_messages_for_processing(self, raw_limit=1000):

        session = Session(bind=self._engine)

        query = session.query(RawMessage, RawMessageProcessingQueue)\
            .filter(RawMessageProcessingQueue.processed_data_id == None)
        query = query.join(RawMessage, RawMessage.id == RawMessageProcessingQueue.raw_chat_data_id)
        query = query.limit(raw_limit)

        res = pd.read_sql(query.statement, session.bind)

        session.close()

        return res

    def write_processed_messages(self, messages_df, raw_data_temp_ids):

        session = Session(bind=self._engine)

        # metadata = sqlalchemy.schema.MetaData(self._engine, reflect=True)
        # table = sqlalchemy.Table(messages_df, metadata, autoload=True)

        # conn = self._engine.connect()
        # table = sqlalchemy.Table('processed_chat_data', Base.metadata, autoreload=True)
        # result = conn.execute(table.insert(), messages_df.to_dict(orient='records'))

        processed_messages = [ProcessedMessage(**pm) for pm in messages_df.to_dict(orient='records')]

        session.add_all(processed_messages)

        session.flush()

        saved_dfs = [pm.to_dict() for pm in processed_messages]

        processing_q_blank = pd.merge(raw_data_temp_ids, pd.DataFrame(data=saved_dfs), on=['temp_id'])[
            ['raw_chat_data_id', 'id']].copy()

        for r in processing_q_blank.to_dict(orient='records'):
            session.query(RawMessageProcessingQueue).filter(RawMessageProcessingQueue.raw_chat_data_id == r['raw_chat_data_id']).update({'processed_data_id': r['id']})

        session.commit()

        session.close()

    def init_message_processing_queue(self):

        session = Session(bind=self._engine)

        session.query(RawMessageProcessingQueue).delete()
        session.query(ProcessedMessage).delete()

        session.flush()

        raw_message_ids = session.query(RawMessage.id).all()

        for raw_message in raw_message_ids:
            self.put_message_to_processing(session, raw_message.id)

        session.commit()
        session.close()

    def get_users(self):

        session = Session(bind=self._engine)

        query = session.query(User.id, User.user_id)

        res = pd.read_sql(query.statement, session.bind)

        session.close()

        return res

    def get_user_id_by_message_id(self, message_id):

        session = Session(bind=self._engine)

        res = session.query(RawMessage.user_id).filter(RawMessage.message_id == message_id).first()
        session.close()

        if res:
            return res.user_id
        else:
            return 0

    def get_all_processed_texts(self, date_limit=None):

        session = Session(bind=self._engine)

        query = session.query(ProcessedMessage.message_text_processed, RawMessage.user_id)
        query = query.join(RawMessageProcessingQueue,
                           ProcessedMessage.id == RawMessageProcessingQueue.processed_data_id)
        query = query.join(RawMessage,
                           RawMessageProcessingQueue.raw_chat_data_id == RawMessage.id)
        query = query.group_by(ProcessedMessage.message_text_processed, RawMessage.user_id)

        res = pd.read_sql(query.statement, session.bind)

        session.close()

        return res

    def get_all_processed_data(self, date_limit=None):

        session = Session(bind=self._engine)

        query = session.query(ProcessedMessage, RawMessage.user_id, func.max(RawMessage.date_time).label("date_time"))
        query = query.join(RawMessageProcessingQueue,
                           ProcessedMessage.id == RawMessageProcessingQueue.processed_data_id)
        query = query.join(RawMessage,
                           RawMessageProcessingQueue.raw_chat_data_id == RawMessage.id)
        query = query.group_by(ProcessedMessage, RawMessage.user_id)

        res = pd.read_sql(query.statement, session.bind)

        session.close()

        return res

    # Typo dictionary communication {

    def get_word_from_typo_d(self, word):

        session = Session(bind=self._engine)

        res = session.query(TypoDictionary).filter(TypoDictionary.word == word).first()

        session.close()
        if res:
            return res.correct_word
        else:
            return 0

    def put_word_to_typo_d(self, word, correct_word, check=0):

        session = Session(bind=self._engine)

        res = session.query(TypoDictionary).filter(TypoDictionary.word == word).first()

        if res:
            session.query(TypoDictionary).filter(
                TypoDictionary.id == res.id).update({'correct_word': correct_word, 'check': check})
        else:
            new_word = TypoDictionary(word=word, correct_word=correct_word, check=check)
            session.add(new_word)

        session.commit()
        session.close()

    # } Typo dictionary communication
