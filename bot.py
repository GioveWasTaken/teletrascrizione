import os
import whisper
import threading
import queue
import re
import subprocess
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

socket.setdefaulttimeout(1000)  # Aumentato il timeout a 1000 secondi

TOKEN = "7837262453:AAGf5poQab9t3v7TGHnn7fGIX8BBtuo6f8k"

# Funzione per scaricare il modello
def download_model(model_size):
    urls = {
        "huggingface": f"https://huggingface.co/openai/whisper-{model_size}/resolve/main/{model_size}.pt",
        "openai": f"https://cdn.openai.com/whisper/models/{model_size}.pt"
    }

    model_path = os.path.expanduser(f"~/.cache/whisper/{model_size}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for source, url in urls.items():
        try:
            logger.info(f"Scaricamento del modello '{model_size}' da {source.capitalize()} ({url})...")
            logger.info(f"📥 Il modello verrà scaricato in: {model_path}")  # Messaggio esplicito sul percorso di download
            response = requests.get(url, stream=True)
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

            logger.info(f"✅ Modello '{model_size}' scaricato con successo da {source.capitalize()} in {model_path}")
            return model_path
        except Exception as e:
            logger.warning(f"⚠️ Errore durante il download da {source.capitalize()}: {e}")

    raise Exception("❌ Impossibile scaricare il modello da entrambe le fonti.")

# Funzione per determinare il modello basato sulle specifiche del sistema
def select_model():
    cpu_count = psutil.cpu_count(logical=True)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    system_info = platform.uname()

    # Disabilitare MPS e forzare l'uso della CPU su Apple Silicon
    is_apple_silicon = "arm" in system_info.machine or "Apple" in system_info.processor
    device = "cuda" if torch.cuda.is_available() and not is_apple_silicon else "cpu"

    # Selezione del modello basata sulle risorse
    if is_apple_silicon:
        model_size = "small" if total_ram < 16 else "medium"  # L'M2 Pro può gestire modelli più grandi
        fp16 = False  # FP16 disabilitato su Apple Silicon
    elif torch.cuda.is_available():
        model_size = "small" if total_ram < 8 else "medium"
        fp16 = True   # FP16 abilitato se disponibile su GPU
    elif cpu_count >= 4 and total_ram >= 8:
        model_size = "medium"
        fp16 = False
    else:
        model_size = "base"
        fp16 = False

    model_path = os.path.expanduser(f"~/.cache/whisper/{model_size}.pt")

    try:
        logger.info(f"Caricamento del modello '{model_size}' dalla cache locale...")
        return whisper.load_model(model_size, device=device), fp16
    except Exception as e:
        logger.warning(f"⚠️ Errore durante il caricamento dalla cache: {e}")
        logger.info("Tentativo di scaricare il modello...")

    model_path = download_model(model_size)
    try:
        return whisper.load_model(model_size, device=device), fp16
    except Exception as e:
        logger.warning(f"⚠️ Errore durante il caricamento del modello scaricato: {e}")
        raise

model, fp16 = select_model()

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

        # Miglioramento della qualità audio con FFmpeg
        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ar", "44100",               # Aumenta il sample rate per maggiore chiarezza
            "-ac", "1",                   # Audio mono per ottimizzazione
            "-af", "dynaudnorm=f=150:g=15,highpass=f=150,lowpass=f=4000,acompressor=threshold=-20dB:ratio=4:attack=5:release=50",
            converted_path
        ], check=True)

        result = model.transcribe(
            converted_path,
            language='it',
            fp16=fp16,  # FP16 abilitato solo se supportato
            condition_on_previous_text=False,
            task="transcribe",
            beam_size=1,               # Ridotto il beam size per maggiore velocità
            temperature=0.0            # Riduce la variabilità per elaborazione più rapida
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

def main():
    logger.info("✅ Avvio del bot...")
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.voice, voice_handler))
    dp.add_handler(CommandHandler("censura", add_censored_word))
    dp.add_handler(CommandHandler("rimuovi_censura", remove_censored_word))
    dp.add_handler(CommandHandler("lista_censure", list_censored_words))

    threading.Thread(target=worker, daemon=True).start()

    updater.start_polling()
    logger.info("🚀 Il bot è attivo e in ascolto!")
    updater.idle()

if __name__ == '__main__':
    main()