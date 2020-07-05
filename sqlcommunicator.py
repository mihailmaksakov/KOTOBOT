import mysql.connector

# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import relationship
#
# Base = declarative_base()
#
# from sqlalchemy import Column, Integer, BINARY, String, ForeignKey, BIGINT, DATETIME
# from sqlalchemy.dialects import mysql
#
#
# class User(Base):
#
#     __tablename__ = 'users'
#
#     id = Column(Integer, primary_key=True)
#     user_id = Column(Integer, index=True)
#     name = Column(String)
#     sex = Column(BINARY)
#
#     def __repr__(self):
#         return f'<User({self.name}, {self.user_id})>'
#
#
# class RawMessage(Base):
#
#     __tablename__ = 'raw_chat_data'
#
#     id = Column(Integer, primary_key=True)
#     user_id = Column(Integer, ForeignKey('users.user_id'))
#     chat_id = Column(mysql.BIGINT)
#     date_time = Column(DATETIME)
#     message_text = Column(mysql.MEDIUMTEXT)
#
#     user = relationship("User", backref="messages")
#
#     name = Column(String)
#     fullname = Column(String)
#
#     def __repr__(self):
#         return "<User(%r, %r)>" % (
#             self.name, self.fullname
#         )


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
        self.__host = host
        self.__database = database
        self.__user = user
        self.__password = password
        self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(host=self.__host, database=self.__database,
                                                      user=self.__user, password=self.__password)
        except mysql.connector.Error as e:
            raise SQLConnectionError(self.__host, self.__database, self.__user, str(e)) from e

    def is_connected(self):
        if self.connection:
            return self.connection.is_connected()
        else:
            return False

    def disconnect(self):
        if self.is_connected():
            self.connection.close()
            self.connection = None
        else:
            self.connection = None

    def stay_connected(self):
        if not self.is_connected():
            self.connect()
        else:
            try:
                self.connection.ping(reconnect=True, attempts=3, delay=5)
            except mysql.connector.Error as e:
                self.disconnect()
                raise SQLConnectionError(self.__host, self.__database, self.__user, str(e)) from e

    def execute_query(self, query_text, params=None, return_result=False, multi=False):

        result = {'rowcount': 0, 'records': None}

        self.stay_connected()

        try:
            cursor = self.connection.cursor(buffered=True)
            if params:
                multi_cursors = cursor.execute(query_text, params, multi=multi)
            else:
                multi_cursors = cursor.execute(query_text, multi=multi)
            self.connection.commit()
            if return_result:
                if multi:
                    multi_cursors = list(multi_cursors)
                    cursor = multi_cursors[len(multi_cursors) - 1]
                records = cursor.fetchall()
                result['rowcount'] = cursor.rowcount
                result['records'] = records
                result['columns'] = cursor.column_names
            cursor.close()
        except mysql.connector.Error as e:
            raise SQLOperationError(query_text, str(e)) from e
            # raise Exception(f'error executing SQL query \n{query_text}\n{e}') from e

        return result

    def put_message(self, user_id, chat_id, date_time, message_id, message_text, reply_to_message_id, sticker,
                    has_photo=0, photo_binary=None):

        insert_query = '''
            INSERT INTO raw_chat_data 
                        (user_id, chat_id, date_time, message_id, message_text, reply_to_message_id, sticker, has_photo, photo)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            select last_insert_id();
        '''

        result = self.execute_query(
            insert_query,
            (
                user_id, chat_id, str(date_time), message_id,
                message_text, reply_to_message_id, sticker, has_photo, photo_binary
            ),
            True, True
        )

        self.put_message_to_processing(result['records'][0][0])

    def put_message_to_processing(self, message_id):

        insert_query = '''
            INSERT INTO raw_data_processing 
                        (raw_chat_data_id)
            VALUES 
                (%s)
        '''

        self.execute_query(
            insert_query,
            (
                message_id,
            )
        )

    def update_users(self, telegram_user):

        query = f"select id from users where user_id = {telegram_user.id}"
        result = self.execute_query(query, None, True)

        if not result['rowcount']:
            insert_query = "INSERT INTO users (name, user_id)\
                               VALUES \
                               (%s, %d)"
            self.execute_query(insert_query, (telegram_user.name, telegram_user.id))

    def prepare_raw_messages_for_processing(self):

        query = '''
            select raw_chat_data.id, raw_chat_data.user_id, raw_chat_data.chat_id,
                raw_chat_data.date_time, 
                ifnull(raw_chat_data.message_text, "") as message_text, 
                raw_chat_data.message_id,
                raw_chat_data.sticker, raw_chat_data.has_photo, raw_chat_data.photo,
                raw_chat_data.reply_to_message_id,
                reply_chat_data.user_id as reply_user_id,
                reply_chat_data.date_time as reply_date_time,
                ifnull(reply_chat_data.message_text, "") as reply_message_text,
                reply_chat_data.sticker as reply_sticker,
                reply_chat_data.has_photo as reply_has_photo,
                reply_chat_data.photo as reply_photo 
            from raw_chat_data as raw_chat_data
                inner join raw_data_processing as raw_data_processing
                    on raw_chat_data.id = raw_data_processing.raw_chat_data_id
                left join raw_chat_data as reply_chat_data
                    on raw_chat_data.reply_to_message_id = reply_chat_data.message_id
            where raw_data_processing.processed_data_id is null
            order by raw_chat_data.chat_id, raw_chat_data.date_time, raw_chat_data.id
        '''
        result = self.execute_query(query, None, True)

        if result['rowcount']:
            return result
        else:
            return None

    def get_user_id_by_message_id(self, message_id):
        query = '''
            select raw_chat_data.user_id 
            from raw_chat_data as raw_chat_data
            where raw_chat_data.message_id = (%s)
        '''
        result = self.execute_query(query, (message_id, ), True)

        if result['rowcount']:
            if result['rowcount'] > 0:
                return result['records'][0][0]

        return None

    def get_all_processed_texts(self, date_limit=None):
        query = '''
             select
             processed_chat_data.message_text_processed, 
             raw_chat_data.user_id 
             from processed_chat_data as processed_chat_data
                inner join raw_data_processing as raw_data_processing
                    on processed_chat_data.id = raw_data_processing.processed_data_id
                    inner join raw_chat_data as raw_chat_data
                        on raw_data_processing.raw_chat_data_id = raw_chat_data.id
         '''
        result = self.execute_query(query, None, True)

        if result['rowcount']:
            if result['rowcount'] > 0:
                return result['records']

        return None
