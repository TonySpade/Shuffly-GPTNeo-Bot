from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Загрузка модели GPT-Neo и токенизатора
model_name = "EleutherAI/gpt-neo-1.3B"  # Используем модель на 1.3 миллиарда параметров
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Устанавливаем pad_token, если он не задан
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Функция для обработки команды /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет! Я бот с GPT-Neo. Напишите мне что-нибудь, и я продолжу.')

# Функция для генерации текста с помощью GPT-Neo
def generate_text(input_text):
    # Токенизация с созданием attention_mask
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Генерация текста
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=255,
        pad_token_id=tokenizer.pad_token_id
    )

    # Декодируем результат
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Функция для обработки текстовых сообщений
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text  # Получаем текст от пользователя
    generated_text = generate_text(user_input)  # Генерируем ответ
    await update.message.reply_text(generated_text)  # Отправляем ответ

def main() -> None:
    # Вставьте сюда ваш токен Telegram бота
    token = "8165087834:AAHZLnOdHLA6A85RbFRE9gLuSgd6zzv5Nmc"

    # Создаем Application
    application = Application.builder().token(token).build()

    # Регистрируем обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # Регистрируем обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    application.run_polling()

if __name__ == "__main__":
    main()
