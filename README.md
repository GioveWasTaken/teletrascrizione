# Telegram Whisper Bot

This is a Telegram bot that transcribes voice messages using OpenAI's Whisper model. It also includes features for censoring specific words, changing transcription language, and toggling transcription time display.

---

## Features

- **Voice Message Transcription:** Converts voice messages to text using OpenAI's Whisper model.
- **Blacklist Functionality:** Admins can add or remove words to be censored from transcriptions.
- **Language Support:** Change transcription language between English (`en`) and Italian (`it`) using `/language`.
- **Toggle Transcription Time:** Enable or disable the display of transcription time using `/time`.
- **Ping Command:** Use `/ping` to check if the bot is online (replies with "pong").

---

## Commands

- `/ping`: Check if the bot is online.
- `/blacklist <word>`: Add a word to the blacklist (admin only).
- `/remove_blacklist <word>`: Remove a word from the blacklist (admin only).
- `/list_blacklist`: Display the list of blacklisted words.
- `/time`: Toggle transcription time display.
- `/language <en|it>`: Change the transcription language.

---

## Requirements

- **Python 3.10**
- **Required Libraries** (included in `requirements.txt`):
  - `torch`
  - `whisper`
  - `tqdm`
  - `python-telegram-bot`
  - `requests`
  - `psutil`

---

## Installation

### Clone the repository:
```sh
git clone https://github.com/GioveWasTaken/teletrascrizione.git
cd teletrascrizione

Create a virtual environment and activate it:

python3.10 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
