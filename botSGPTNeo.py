from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# �������� ������ GPT-Neo � ������������
model_name = "EleutherAI/gpt-neo-2.7B"  # ���������� ������ �� 2.7 ��������� ����������
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# ������� ��� ��������� ������� /start
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('������! � ��� � GPT-Neo. �������� ��� ���-������, � � ��������.')

# ������� ��� ��������� ������ � ������� GPT-Neo
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, do_sample=True, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ������� ��� ��������� ��������� ���������
def handle_message(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    response = generate_text(user_text)
    update.message.reply_text(response)

def main() -> None:
    # �������� ���� ��� ����� Telegram ����
    token = "8165087834:AAHZLnOdHLA6A85RbFRE9gLuSgd6zzv5Nmc"
    updater = Updater(token)

    # �������� ��������� ��� ����������� ������������
    dispatcher = updater.dispatcher

    # ������������ ���������� ������� /start
    dispatcher.add_handler(CommandHandler("start", start))

    # ������������ ���������� ��������� ���������
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # ��������� ����
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()