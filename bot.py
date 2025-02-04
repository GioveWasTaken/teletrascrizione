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
import gc  # Import per la gestione della memoria
import time  # Per misurare i tempi di esecuzione

# Configurazione logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

socket.setdefaulttimeout(1000)  # Aumentato il timeout a 1000 secondi

# Ottimizzazione: Forzare l'uso di tutti i core disponibili
torch.set_num_threads(psutil.cpu_count(logical=True) * 2)  # Massimizza l'uso dei core disponibili

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

# Forzare sempre l'utilizzo del modello 'small'
def select_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = True if device == "cuda" else False
    model_size = "small"

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
                transcription, transcription_time = transcribe_audio(file_path)
                if transcription:
                    transcription = censor_text(transcription)
                    update.message.reply_text(f"📝 Trascrizione: {transcription}\n⏱️ _Tempo di trascrizione: {transcription_time:.2f} secondi_", parse_mode='Markdown')
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

        # Ottimizzazione FFmpeg per ridurre i tempi e migliorare la qualità
        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ar", "16000",               # Riduzione della frequenza di campionamento per maggiore velocità
            "-ac", "1",                   # Audio mono
            "-filter:a", "volume=1.2",    # Leggera amplificazione per migliorare la chiarezza
            converted_path
        ], check=True)

        start_time = time.time()  # Inizio misurazione del tempo di trascrizione

        result = model.transcribe(
            converted_path,
            language='it',
            fp16=fp16,
            condition_on_previous_text=False,
            task="transcribe",
            beam_size=1,               # Ridotto per aumentare la velocità
            temperature=0.0
        )

        transcription_time = time.time() - start_time
        logger.info(f"⏱️ Tempo di trascrizione: {transcription_time:.2f} secondi")

        text = result.get('text', '').strip()

        if len(text) == 0 or text.isspace():
            return None, transcription_time

        return text, transcription_time
    except subprocess.CalledProcessError as e:
        logger.exception("Errore di conversione audio con ffmpeg:")
    except Exception as e:
        logger.exception("Errore durante la trascrizione:")
    finally:
        if os.path.exists(converted_path):
            os.remove(converted_path)
        gc.collect()  # Libera la memoria
        torch.cuda.empty_cache()  # Pulisce la cache di PyTorch (anche su CPU)

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
