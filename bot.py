import os
import whisper
import threading
import queue
import re
import subprocess
import tiktoken  # Importazione di tiktoken per una migliore gestione del token
import psutil  # Per rilevare le specifiche del sistema
import platform
import torch
from telegram import Update, ChatMember
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler
import socket
socket.setdefaulttimeout(500)  # Timeout aumentato a 5 minuti

# Token del bot Telegram
TOKEN = "7837262453:AAGf5poQab9t3v7TGHnn7fGIX8BBtuo6f8k" #Assicurati che la variabile sia definita o sostituiscila con il token diretto

# Funzione per determinare il modello in base alle specifiche del sistema
def select_model():
    cpu_count = psutil.cpu_count(logical=True)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # Convertito in GB
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    system_info = platform.uname()

    # Rilevamento specifico per MacBook M1/M2
    if "arm" in system_info.machine or "Apple" in system_info.processor or "M1" in system_info.processor or "M2" in system_info.processor:
        print("MacBook Apple Silicon rilevato. Utilizzo di 'medium' con MPS.")
        return whisper.load_model("medium").to(device)

    if device != "cpu":
        print(f"Dispositivo accelerato rilevato: {device}")
        return whisper.load_model("medium").to(device)  # Usa 'medium' su GPU/MPS
    elif cpu_count >= 4 and total_ram >= 8:
        print("Sistema con alte prestazioni rilevato. Caricamento modello 'small'...")
        return whisper.load_model("small")
    else:
        print("Sistema a basse risorse rilevato. Caricamento modello 'base'...")
        return whisper.load_model("base")

# Caricamento del modello basato sulle specifiche del sistema
model = select_model()

# Coda per gestire i messaggi
audio_queue = queue.Queue()

# Dizionario per le parole da censurare
censored_words = set()

# Funzione di elaborazione asincrona
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
                update.message.reply_text("Errore: file audio non trovato.")
        except Exception as e:
            update.message.reply_text(f"Errore nella trascrizione: {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            audio_queue.task_done()

# Funzione per trascrivere l'audio
def transcribe_audio(file_path):
    try:
        # Conversione dell'audio per migliorare la velocità
        converted_path = file_path.replace(".ogg", ".wav")
        subprocess.run([
            "ffmpeg", "-i", file_path, "-ar", "8000", "-ac", "1", converted_path
        ], check=True)

        result = model.transcribe(
            converted_path,
            language='it',
            fp16=False,               # Disattivazione FP16 per CPU
            condition_on_previous_text=False,  # Migliora la velocità disattivando la dipendenza dal testo precedente
            task="transcribe",        # Esplicita il tipo di task
            beam_size=5               # Ottimizzazione della decodifica per una migliore velocità
        )
        text = result.get('text', '').strip()

        # Verifica se il testo contiene parole
        if len(text) == 0 or text.isspace():
            return None

        return text
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Errore di conversione audio con ffmpeg: {e}")
    except Exception as e:
        raise RuntimeError(f"Errore durante la trascrizione: {str(e)}")
    finally:
        if os.path.exists(converted_path):
            os.remove(converted_path)

# Funzione per censurare il testo
def censor_text(text):
    for word in censored_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub('[*****]', text)
    return text

# Funzione per gestire i messaggi vocali
def voice_handler(update: Update, context: CallbackContext):
    file = update.message.voice.get_file()
    file_path = f"{file.file_id}.ogg"
    file.download(file_path)

    # Aggiungi il messaggio alla coda
    audio_queue.put((update, file_path))

# Comando per aggiungere parole alla lista di censura
def add_censored_word(update: Update, context: CallbackContext):
    user = update.effective_user
    chat = update.effective_chat

    # Verifica se l'utente è un amministratore
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

# Comando per rimuovere parole dalla lista di censura
def remove_censored_word(update: Update, context: CallbackContext):
    user = update.effective_user
    chat = update.effective_chat

    # Verifica se l'utente è un amministratore
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

# Comando per visualizzare le parole censurate
def list_censored_words(update: Update, context: CallbackContext):
    if censored_words:
        update.message.reply_text("Parole censurate:\n" + "\n".join(censored_words))
    else:
        update.message.reply_text("Nessuna parola censurata.")

# Funzione principale
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.voice, voice_handler))
    dp.add_handler(CommandHandler("censura", add_censored_word))
    dp.add_handler(CommandHandler("rimuovi_censura", remove_censored_word))
    dp.add_handler(CommandHandler("lista_censure", list_censored_words))

    # Avvia 1 worker per ridurre il carico sulla CPU
    threading.Thread(target=worker, daemon=True).start()

    updater.start_polling()
    updater.idle()

    # Ferma i worker alla chiusura
    audio_queue.put((None, None))

if __name__ == '__main__':
    main()
