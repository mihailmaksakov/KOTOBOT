from flickrapi import FlickrAPI
import random

from telegram import Update
from telegram.ext import CallbackContext


class ImageDownloader:

    def __init__(self, flickr_public, flickr_secret):
        self.flickr_public = flickr_public
        self.flickr_secret = flickr_secret

    def get_flickr_image(self, search_str):
        # extras = 'url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'
        extras = 'url_sq,url_t,url_s,url_q,url_m'
        random.seed
        FLICKR = FlickrAPI(self.flickr_public, self.flickr_secret, format='parsed-json')
        data = FLICKR.photos.search(text=search_str, per_page=1, sort='interestingness-desc', safe_search=3)
        if data['stat'] == 'ok':
            photo_data = FLICKR.photos.search(text=search_str, per_page=1, page=random.randint(1, min(int(data['photos']['total']), 1000)), extras=extras, sort='interestingness-desc')
            if photo_data['photos']['photo']:
                extras_array = extras.split(',')
                for extra in extras.split(',')[::-1]:
                    return photo_data['photos']['photo'][0][extra]

    def get_kitten(self):
        return self.get_flickr_image('funny cat')

    def get_tyan(self):
        return self.get_flickr_image(get_random_grl_search())

    def get_elmacho(self):
        return self.get_flickr_image('handsome man portrait')

    def koto_handler(self, update: Update, context: CallbackContext):
        context.bot.send_photo(chat_id=update.message.chat_id, photo=self.get_kitten())
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=f'/help /koto /macho /chick')

    def macho_handler(self, update: Update, context: CallbackContext):
        context.bot.send_photo(chat_id=update.message.chat_id, photo=self.get_elmacho())
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=f'/help /koto /macho /chick')

    def chick_handler(self, update: Update, context: CallbackContext):
        context.bot.send_photo(chat_id=update.message.chat_id, photo=self.get_tyan())
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=f'/help /koto /macho /chick')


def get_random_grl_search():
    keywords = [
        'beautiful woman portrait',
        'sexy girl nu',
        'sexy girl portrait',
        'sexy girl',
        'sexy woman',
        'naked woman',
        'naked nu woman',
        'nu woman',
        'beautiful girl',
        'naked tits',
    ]
    random.seed
    return random.choice(keywords)
