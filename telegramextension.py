from telegram.ext import (Filters, Handler)
from telegram import Update


class DefaultTelegramHandler(Handler):

    def __init__(self,
                 callback,
                 filters=None,
                 pass_args=False,
                 pass_update_queue=False,
                 pass_job_queue=False,
                 pass_user_data=False,
                 pass_chat_data=False):
        super(DefaultTelegramHandler, self).__init__(
            callback,
            pass_update_queue=pass_update_queue,
            pass_job_queue=pass_job_queue,
            pass_user_data=pass_user_data,
            pass_chat_data=pass_chat_data)

        if filters:
            self.filters = Filters.update.messages & filters
        else:
            self.filters = Filters.update.messages

        self.pass_args = pass_args

    def check_update(self, update):
        if isinstance(update, Update) and update.effective_message:
            return self.filters(update)
