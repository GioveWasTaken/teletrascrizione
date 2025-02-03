# Base image
FROM python:3.10-slim

# Installazione di ffmpeg e dipendenze di sistema
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Copia dei file del progetto
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# Variabile d'ambiente per il token del bot
ENV TELEGRAM_BOT_TOKEN=7837262453:AAGf5poQab9t3v7TGHnn7fGIX8BBtuo6f8k

# Avvio del bot
CMD ["python", "bot.py"]
