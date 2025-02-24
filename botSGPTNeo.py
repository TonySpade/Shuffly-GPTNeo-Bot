from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Загрузка модели GPT-Neo и токенизатора
model_name = "EleutherAI/gpt-neo-2.7B"  # Используем модель на 2.7 миллиарда параметров
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Функция для обработки команды /start
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я бот с GPT-Neo. Напишите мне что-нибудь, и я продолжу.')

# Функция для генерации текста с помощью GPT-Neo
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, do_sample=True, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Функция для обработки текстовых сообщений
def handle_message(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    response = generate_text(user_text)
    update.message.reply_text(response)

def main() -> None:
    # Вставьте сюда ваш токен Telegram бота
    token = "8165087834:AAHZLnOdHLA6A85RbFRE9gLuSgd6zzv5Nmc"
    updater = Updater(token)

    # Получаем диспетчер для регистрации обработчиков
    dispatcher = updater.dispatcher

    # Регистрируем обработчик команды /start
    dispatcher.add_handler(CommandHandler("start", start))

    # Регистрируем обработчик текстовых сообщений
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Запускаем бота
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()