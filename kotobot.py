import argparse
import logging
import os
# import random
import signal
# import time
# import re
from datetime import timedelta
from time import sleep, time, ctime

from sqlcommunicator import SQLCommunicator
from sqlcommunicator import Error as SQLCommunicatorError
from imagedownloader import ImageDownloader

from telegram import Update
from telegram import PhotoSize
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          CallbackContext)
from telegramextension import DefaultTelegramHandler

import pandas as pd

import jobs

import message_processing as mp
import image_processing as ip
from userdetector import UserDetector

logging.basicConfig(filename="kotobot.log", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

jobs_array = []

STICKERS = {
    'UR_SO_FUN_GERMAN': 'CAACAgIAAxkBAAPYXmSx4v-NBqcFIxlw2VTpj84JcI4AAicFAAKAPwcodEkpXQABag8aGAQ',
    'DESPISE_DOGO':     'CAACAgEAAxkBAAPZXmSyMsmI-mB3lKAyra3JWHegp_wAAqcQAAKZf4gCAAGYtdndv2XFGAQ',
    }

SELF_NAME = '@kotoboto_bot'

MAX_IMAGE_SIZE = pow(2, 24)

sql_communicator = SQLCommunicator('localhost', 'kotoboto', 'root')

SIG_EXIT = 0
NEW_MESSAGES_PROCESSING_PERIOD = 60*60*6
USER_SIMILARITY_MODEL_UDPATE_PERIOD = 60*60*6


def is_roman(user):
    return user.id == 829112612


def is_mike(user):
    return user.id == 246236811


def is_anton(user):
    return user.id == 41067949


def message_handler(update: Update, context: CallbackContext):

    try:
        sql_communicator.update_users(update.effective_user)
    except SQLCommunicatorError as e:
        logger.error(e)

    has_photo = 0
    photo_binary = None
    doc_to_download = None
    if update.effective_message.document and 'image' in update.effective_message.document.mime_type.lower():
        doc_to_download = update.effective_message.document.get_file()
    elif isinstance(update.effective_message.effective_attachment, list) \
            and len(update.effective_message.effective_attachment) > 0:
        if isinstance(update.effective_message.effective_attachment[0], PhotoSize):
            doc_to_download = update.effective_message.effective_attachment[0].get_file()
    if doc_to_download:
        f_name = doc_to_download.download(f'./tmp/{doc_to_download.file_id}')
        if os.path.getsize(f_name) <= MAX_IMAGE_SIZE:
            with open(f_name, 'rb') as file:
                has_photo = 1
                photo_binary = file.read()
            os.remove(f_name)
    try:
        sql_communicator.put_message(
            update.effective_user.id, update.effective_chat.id, update.effective_message.date,
            update.effective_message.message_id, update.effective_message.text_html,
            update.effective_message.reply_to_message.message_id if update.effective_message.reply_to_message else 0,
            update.effective_message.sticker.file_id if update.effective_message.sticker else None,
            has_photo, photo_binary
        )
    except SQLCommunicatorError as e:
        logger.error(e)


def help_handler(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=f'/help /koto /macho /chick',
                             reply_to_message_id=update.message.message_id)


def che(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=f'чёбля? /help /koto /macho /chick',
                             reply_to_message_id=update.message.message_id)


# def photo(update: Update, context: CallbackContext):
#
#     if context.match.groups()[0]:
#         gotName = context.match.groups()[0].strip()
#         suffix = ''
#         if gotName[-1] in 'аеуэюь':
#             gotName = gotName[:-1]
#             suffix = 'и'
#         if gotName[-1] in 'о':
#             gotName = gotName[:-1]
#             suffix = 'а'
#         elif gotName[-1] in 'бвгджзклмнрстфхчшщ':
#             suffix = 'а'
#
#         context.bot.send_message(chat_id=update.message.chat_id,
#                                  text=f'от {gotName}{suffix} слышу',
#                                  reply_to_message_id=update.message.message_id)
#
#     random.seed
#     rnd = random.randint(1, 2)
#     if is_roman(update.effective_user) and rnd == 2:
#         url = get_elmacho()
#     else:
#         if re.match(r'кот|кош|кыц|киц|кис', context.match.groups()[2].strip()):
#             # print(update.effective_user.id, 1)
#             url = get_kitten()
#         elif re.match(r'тян|гёрл|дев|жен', context.match.groups()[2].strip()):
#             if is_mike(update.effective_user):
#                 # print(update.effective_user.id, 3)
#                 url = get_sx_tyan()
#             else:
#                 print(update.effective_user.id, 4)
#                 url = get_tyan()
#         elif re.match(r'муж|мальч|мачо|паца', context.match.groups()[2].strip()):
#             # print(update.effective_user.id, 2)
#             url = get_elmacho()
#         else:
#             context.bot.send_message(chat_id=update.message.chat_id,
#                                      text=f'сасад /help',
#                                      reply_to_message_id=update.message.message_id)
#             return
#     chat_id = update.message.chat_id
#     context.bot.send_photo(chat_id=chat_id, photo=url)
#     context.bot.send_message(chat_id=update.message.chat_id,
#                              text=f'/help')


def error_handler(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def on_stop(*args):
    global SIG_EXIT
    SIG_EXIT = 1
    # os.exit(0)


def process_messages_job():

    sql_communicator_job = SQLCommunicator('localhost', 'kotoboto', 'root')

    # select uprocessed messages from raw_data_processing table
    try:
        data = sql_communicator_job.prepare_raw_messages_for_processing(100)
    except SQLCommunicatorError as e:
        logger.error(e)
        return

    if not data.empty:

        # data = pd.DataFrame(result['records'])
        # data.columns = result['columns']

        temp_aggregated_data_indeces = []

        # collect whole message text from message queue based on user id
        aggregated_data_list = []

        prev_chat_user = 0
        has_sticker = False
        has_photo = False
        reply_message = 0

        for index, raw_message in data.iterrows():

            current_chat_user = raw_message['chat_id'] + raw_message['user_id']
            current_has_sticker = not raw_message['sticker'] is None and not raw_message['sticker'] == ''
            current_has_photo = raw_message['has_photo'] == 1
            current_reply_message = raw_message['reply_to_message_id']

            new_message = not prev_chat_user == current_chat_user
            new_message = new_message or has_sticker and current_has_sticker
            new_message = new_message or has_photo and current_has_photo
            new_message = new_message or (not reply_message == 0 and not reply_message == current_reply_message)

            has_sticker = has_sticker or current_has_sticker
            has_photo = has_photo or current_has_photo
            reply_message = reply_message if reply_message else current_reply_message

            if not new_message:
                last_row = aggregated_data_list[-1]
                if raw_message['message_text']:
                    last_row['message_text'] += '\n' + str(raw_message['message_text'])
            else:
                aggregated_data_list.append(raw_message.to_dict())
                last_row = aggregated_data_list[-1]
                last_row['message_text'] = mp.raw_message_to_string(last_row['message_text'])
                has_sticker = False
                has_photo = False
                reply_message = 0

            temp_aggregated_data_indeces.append(len(aggregated_data_list) - 1)

            prev_chat_user = current_chat_user

        aggregated_data = pd.DataFrame(data=aggregated_data_list)

        processed_data_indeces = pd.DataFrame({'raw_chat_data_id': data['id'],
                                               'temp_id': pd.Series(temp_aggregated_data_indeces)})

        # fill in analysis criteria for every whole message
        processed_chat_data = []
        for index, single_message_data in aggregated_data.iterrows():

            message_text = single_message_data['message_text'].lower()
            message_recipient = mp.get_message_recipient(message_text, single_message_data['reply_to_message_id'],
                                                         sql_communicator_job)

            # message_text_wo_typos = mp.clean_typos(message_text, sql_communicator_job)
            message_text_processed = mp.normalize_string(message_text, sql_communicator_job)

            processed_chat_data.append({
                'temp_id': index,
                'obscene_terms_count': mp.get_obscene_terms_count(message_text),
                'sentiment': mp.get_sentiment(message_text_processed),
                'other_user_similarity': mp.other_user_similarity(message_text_processed),
                'contains_text': mp.content_is_empty(message_text),
                'contains_sticker': mp.content_is_empty(single_message_data['sticker']),
                'contains_image': single_message_data['has_photo'] == 1,
                'message_recipient': message_recipient,
                'recipient_is_bot': SELF_NAME in message_text,
                'topic': mp.get_message_topic(message_text_processed, sql_communicator_job),
                'image_topic': ip.get_image_topic(single_message_data['has_photo'], single_message_data['photo']),
                'image_adult': ip.image_contains_adult(single_message_data['has_photo'], single_message_data['photo']),
                'message_text': message_text,
                'message_text_processed': message_text_processed
            })

        processed_chat_data_df = pd.DataFrame(data=processed_chat_data)

        sql_communicator_job.write_processed_messages(processed_chat_data_df, processed_data_indeces)

        # sql_communicator_job.disconnect()

    sleep(NEW_MESSAGES_PROCESSING_PERIOD)


def update_user_detector_model():

    UserDetector().update_model(SQLCommunicator('localhost', 'kotoboto', 'root'))

    sleep(USER_SIMILARITY_MODEL_UDPATE_PERIOD)


def get_telegram_proxy_kwargs(args):

    result = {}

    if args.proxy_url:
        result = {
            'proxy_url': args.proxy_url,
            # Optional, if you need authentication:
            'urllib3_proxy_kwargs': {
                'username': args.proxy_user,
                'password': args.proxy_password
            }
        }

    return result


def main(args):

    for sig in (signal.SIGBREAK, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_stop)

    jobs_array.append(jobs.Job(interval=timedelta(seconds=jobs.JOBS_WAIT_TIME_SECONDS), execute=process_messages_job))
    jobs_array.append(jobs.Job(interval=timedelta(seconds=jobs.JOBS_WAIT_TIME_SECONDS), execute=update_user_detector_model))

    for job in jobs_array:
        job.start()

    updater = Updater(args.token, request_kwargs=get_telegram_proxy_kwargs(args), use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    id = ImageDownloader(args.flickr_key, args.flickr_secret)
    # dp.add_handler(MessageHandler(Filters.regex('[сС]лыш[ь]* (.*)(дай|запости|скинь|покажи|скинь|ещё|давай) ([а-я]+)'), photo))
    dp.add_handler(MessageHandler(Filters.regex('(^[сС]лыш[ь]?$)'), che))
    dp.add_handler(CommandHandler('help', help_handler))
    dp.add_handler(CommandHandler('koto', callback=id.koto_handler))
    dp.add_handler(CommandHandler('macho', callback=id.macho_handler))
    dp.add_handler(CommandHandler('chick', callback=id.chick_handler))
    dp.add_handler(DefaultTelegramHandler(callback=message_handler))
    # log all errors
    dp.add_error_handler(error_handler)

    # Start the Bot
    updater.start_polling()

    while not SIG_EXIT:
        sleep(1)

    updater.stop()
    # sql_communicator.disconnect()
    for job in jobs_array:
        job.stop()
    exit(0)


parser = argparse.ArgumentParser(description='Process telegram token.')

parser.add_argument('-t', '--token', type=str, help='telegram token', required=True)
parser.add_argument('-fk', '--flickr_key', type=str, help='flickr public key', required=True)
parser.add_argument('-fs', '--flickr_secret', type=str, help='flickr secret', required=True)
parser.add_argument('-u', '--proxy_url', type=str, help='proxy URL')
parser.add_argument('-s', '--proxy_user', type=str, help='proxy user')
parser.add_argument('-p', '--proxy_password', type=str, help='proxy password')

if __name__ == '__main__':
    main(parser.parse_args())