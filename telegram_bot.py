from telegram.ext import Updater, CommandHandler, ConversationHandler, MessageHandler, Filters
import logging
from twitter_rnn import SentimentPredictor

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="I'm a bot, please talk to me!")





def cancel(update):
    update.message.reply_text('Bye! I hope we can talk again some day.',
                              reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


def main():
    updater = Updater(token='649305193:AAG0qWkAerbGVyFBjYr8gGaoyU41E0Ah14c')
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states= {
            SENTIMENT : [MessageHandler(Filters.sentiment, sentiment)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]

    )


    dispatcher.add_handler(conv_handler)


    updater.start_polling()


if __name__ == '__main__':
    main()