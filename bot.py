import os
import whisper
import threading
import queue
import re
import subprocess
import tiktoken
import psutil
import platform
import torch
import requests
from tqdm import tqdm
from telegram import Update, ChatMember
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler
import socket
import logging

# Configurazione logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

socket.setdefaulttimeout(500)

TOKEN = "7837262453:AAGf5poQab9t3v7TGHnn7fGIX8BBtuo6f8k"

# Funzione per scaricare il modello da Hugging Face
def download_model_from_huggingface(model_size):
    urls = {
        "tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
        "base": "https://huggingface.co/openai/whisper-base/resolve/main/base.pt",
        "small": "https://huggingface.co/openai/whisper-small/resolve/main/small.pt",
        "medium": "https://huggingface.co/openai/whisper-medium/resolve/main/medium.pt",
        "large": "https://huggingface.co/openai/whisper-large/resolve/main/large.pt"
    }

    if model_size not in urls:
        raise ValueError(f"Modello '{model_size}' non supportato.")

    model_url = urls[model_size]
    model_path = os.path.expanduser(f"~/.cache/whisper/{model_size}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    logger.info(f"Scaricamento del modello '{model_size}' da Hugging Face...")
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(model_path, 'wb') as file, tqdm(
        desc=model_size,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

    logger.info(f"✅ Modello '{model_size}' scaricato con successo in {model_path}")
    return model_path

# Funzione per determinare il modello basato sulle specifiche del sistema
def select_model():
    cpu_count = psutil.cpu_count(logical=True)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    system_info = platform.uname()

    if "arm" in system_info.machine or "Apple" in system_info.processor:
        model_size = "medium"
    elif device != "cpu":
        model_size = "medium"
    elif cpu_count >= 4 and total_ram >= 8:
        model_size = "small"
    else:
        model_size = "base"

    try:
        logger.info(f"Caricamento del modello '{model_size}'...")
        return whisper.load_model(model_size).to(device)
    except Exception as e:
        logger.warning(f"⚠️ Errore durante il caricamento del modello '{model_size}': {e}")
        logger.info("Tentativo di scaricare il modello da Hugging Face...")

        model_path = download_model_from_huggingface(model_size)
        return whisper.load_model(model_path).to(device)

model = select_model()

audio_queue = queue.Queue()
censored_words = set()

def worker():
    while True:
        update, file_path = audio_queue.get()
        if update is None:
            break

        try:
            if os.path.exists(file_path):
                transcription = transcribe_audio(file_path)
                if transcription:
                    transcription = censor_text(transcription)
                    update.message.reply_text(f"📝 Trascrizione: {transcription}")
            else:
                logger.error("Errore: file audio non trovato.")
        except Exception as e:
            logger.exception("Errore nella trascrizione:")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            audio_queue.task_done()

def transcribe_audio(file_path):
    try:
        converted_path = file_path.replace(".ogg", ".wav")
        subprocess.run([
            "ffmpeg", "-i", file_path, "-ar", "8000", "-ac", "1", converted_path
        ], check=True)

        result = model.transcribe(
            converted_path,
            language='it',
            fp16=False,
            condition_on_previous_text=False,
            task="transcribe",
            beam_size=5
        )
        text = result.get('text', '').strip()

        if len(text) == 0 or text.isspace():
            return None

        return text
    except subprocess.CalledProcessError as e:
        logger.exception("Errore di conversione audio con ffmpeg:")
    except Exception as e:
        logger.exception("Errore durante la trascrizione:")
    finally:
        if os.path.exists(converted_path):
            os.remove(converted_path)

def censor_text(text):
    for word in censored_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub('[*****]', text)
    return text

def voice_handler(update: Update, context: CallbackContext):
    logger.info(f"Ricevuto messaggio vocale da {update.effective_user.username}")
    file = update.message.voice.get_file()
    file_path = f"{file.file_id}.ogg"
    file.download(file_path)
    audio_queue.put((update, file_path))

def add_censored_word(update: Update, context: CallbackContext):
    user = update.effective_user
    chat = update.effective_chat
    member = chat.get_member(user.id)
    if member.status in [ChatMember.ADMINISTRATOR, ChatMember.CREATOR]:
        if context.args:
            word = " ".join(context.args)
            censored_words.add(word)
            update.message.reply_text(f"Parola '{word}' aggiunta alla lista di censura.")
        else:
            update.message.reply_text("Usa il comando così: /censura <parola>")
    else:
        update.message.reply_text("Solo gli amministratori possono aggiungere parole alla lista di censura.")

def remove_censored_word(update: Update, context: CallbackContext):
    user = update.effective_user
    chat = update.effective_chat
    member = chat.get_member(user.id)
    if member.status in [ChatMember.ADMINISTRATOR, ChatMember.CREATOR]:
        if context.args:
            word = " ".join(context.args)
            if word in censored_words:
                censored_words.remove(word)
                update.message.reply_text(f"Parola '{word}' rimossa dalla lista di censura.")
            else:
                update.message.reply_text(f"La parola '{word}' non è presente nella lista di censura.")
        else:
            update.message.reply_text("Usa il comando così: /rimuovi_censura <parola>")
    else:
        update.message.reply_text("Solo gli amministratori possono rimuovere parole dalla lista di censura.")

def list_censored_words(update: Update, context: CallbackContext):
    if censored_words:
        update.message.reply_text("Parole censurate:\n" + "\n".join(censored_words))
    else:
        update.message.reply_text("Nessuna parola censurata.")

def debug_handler(update: Update, context: CallbackContext):
    logger.info(f"Ricevuto un messaggio da {update.effective_user.username}: {update.message.text}")

def main():
    logger.info("✅ Avvio del bot...")
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.voice, voice_handler))
    dp.add_handler(CommandHandler("censura", add_censored_word))
    dp.add_handler(CommandHandler("rimuovi_censura", remove_censored_word))
    dp.add_handler(CommandHandler("lista_censure", list_censored_words))
    dp.add_handler(MessageHandler(Filters.all, debug_handler))

    threading.Thread(target=worker, daemon=True).start()

    updater.start_polling()
    logger.info("🚀 Il bot è attivo e in ascolto!")
    updater.idle()

    audio_queue.put((None, None))

if __name__ == '__main__':
    main()
