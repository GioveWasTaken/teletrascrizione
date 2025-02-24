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
import gc
import time

# Logging configuration
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

socket.setdefaulttimeout(1000)

# Optimize: Use all available cores
torch.set_num_threads(psutil.cpu_count(logical=True) * 2)

TOKEN = "INSERT TOKEN HERE"

# Default settings
show_transcription_time = False
transcription_language = "en"  # Default language set to English

# Function to download the model
def download_model(model_size):
    urls = {
        "huggingface": f"https://huggingface.co/openai/whisper-{model_size}/resolve/main/{model_size}.pt",
        "openai": f"https://cdn.openai.com/whisper/models/{model_size}.pt"
    }

    model_path = os.path.expanduser(f"~/.cache/whisper/{model_size}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for source, url in urls.items():
        try:
            logger.info(f"Downloading model '{model_size}' from {source.capitalize()} ({url})...")
            logger.info(f"The model will be saved at: {model_path}")
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

            logger.info(f"Model '{model_size}' successfully downloaded from {source.capitalize()} at {model_path}")
            return model_path
        except Exception as e:
            logger.warning(f"Error while downloading from {source.capitalize()}: {e}")

    raise Exception("Unable to download the model from both sources.")

# Always use the 'small' model
def select_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = True if device == "cuda" else False
    model_size = "small"

    model_path = os.path.expanduser(f"~/.cache/whisper/{model_size}.pt")

    try:
        logger.info(f"Loading model '{model_size}' from local cache...")
        return whisper.load_model(model_size, device=device), fp16
    except Exception as e:
        logger.warning(f"Error loading from cache: {e}")
        logger.info("Trying to download the model...")

    model_path = download_model(model_size)
    try:
        return whisper.load_model(model_size, device=device), fp16
    except Exception as e:
        logger.warning(f"Error loading the downloaded model: {e}")
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
                    message = f"Transcription: {transcription}"
                    if show_transcription_time:
                        message += f"\nTranscription time: {transcription_time:.2f} seconds"
                    update.message.reply_text(message, parse_mode='Markdown')
            else:
                logger.error("Error: audio file not found.")
        except Exception as e:
            logger.exception("Error during transcription:")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            audio_queue.task_done()

def transcribe_audio(file_path):
    try:
        converted_path = file_path.replace(".ogg", ".wav")

        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ar", "16000",
            "-ac", "1",
            "-filter:a", "volume=1.2"
        ], check=True)

        start_time = time.time()

        result = model.transcribe(
            converted_path,
            language=transcription_language,
            fp16=fp16,
            condition_on_previous_text=False,
            task="transcribe",
            beam_size=1,
            temperature=0.0
        )

        transcription_time = time.time() - start_time
        logger.info(f"Transcription time: {transcription_time:.2f} seconds")

        text = result.get('text', '').strip()

        if len(text) == 0 or text.isspace():
            return None, transcription_time

        return text, transcription_time
    except subprocess.CalledProcessError as e:
        logger.exception("Audio conversion error with ffmpeg:")
    except Exception as e:
        logger.exception("Error during transcription:")
    finally:
        if os.path.exists(converted_path):
            os.remove(converted_path)
        gc.collect()
        torch.cuda.empty_cache()

def censor_text(text):
    for word in censored_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub('[*****]', text)
    return text

def voice_handler(update: Update, context: CallbackContext):
    logger.info(f"Received voice message from {update.effective_user.username}")
    file = update.message.voice.get_file()
    file_path = f"{file.file_id}.ogg"
    file.download(file_path)
    audio_queue.put((update, file_path))

def add_blacklist_word(update: Update, context: CallbackContext):
    if context.args:
        word = " ".join(context.args)
        censored_words.add(word)
        update.message.reply_text(f"Word '{word}' added to the blacklist.")
    else:
        update.message.reply_text("Use the command as follows: /blacklist <word>")

def remove_blacklist_word(update: Update, context: CallbackContext):
    if context.args:
        word = " ".join(context.args)
        if word in censored_words:
            censored_words.remove(word)
            update.message.reply_text(f"Word '{word}' removed from the blacklist.")
        else:
            update.message.reply_text(f"Word '{word}' is not in the blacklist.")
    else:
        update.message.reply_text("Use the command as follows: /remove_blacklist <word>")

def list_blacklist(update: Update, context: CallbackContext):
    if censored_words:
        update.message.reply_text("Blacklisted words:\n" + "\n".join(censored_words))
    else:
        update.message.reply_text("No words in the blacklist.")


# Command to toggle transcription time display
def toggle_time(update: Update, context: CallbackContext):
    global show_transcription_time
    show_transcription_time = not show_transcription_time
    status = "enabled" if show_transcription_time else "disabled"
    update.message.reply_text(f"Transcription time display is now {status}.")


# Command to change transcription language
def change_language(update: Update, context: CallbackContext):
    global transcription_language
    if context.args:
        lang = context.args[0].lower()
        if lang in ['en', 'it']:
            transcription_language = lang
            update.message.reply_text(f"Transcription language set to '{lang}'.")
        else:
            update.message.reply_text("Invalid language. Use /language en or /language it.")
    else:
        update.message.reply_text("Use the command as follows: /language <en|it>")


# Command to check if the bot is responsive
def ping(update: Update, context: CallbackContext):
    update.message.reply_text("pong")


# Main function to start the bot
def main():
    logger.info("Starting the bot...")
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    # Handlers for voice message transcription
    dp.add_handler(MessageHandler(Filters.voice, voice_handler))

    # Handlers for blacklist management
    dp.add_handler(CommandHandler("blacklist", add_blacklist_word))
    dp.add_handler(CommandHandler("remove_blacklist", remove_blacklist_word))
    dp.add_handler(CommandHandler("list_blacklist", list_blacklist))

    # Handler for toggling transcription time display
    dp.add_handler(CommandHandler("time", toggle_time))

    # Handler for changing transcription language
    dp.add_handler(CommandHandler("language", change_language))

    # Handler for ping command
    dp.add_handler(CommandHandler("ping", ping))

    # Start worker thread
    threading.Thread(target=worker, daemon=True).start()

    # Start the bot
    updater.start_polling()
    logger.info("The bot is now listening...")
    updater.idle()


if __name__ == '__main__':
    main()
