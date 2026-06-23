FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached across rebuilds
# that only change source code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No EXPOSE needed: the bot only makes outbound requests (long polling
# to Telegram, HTTPS calls to Gemini) -- it never listens on a port.
CMD ["python", "bot.py"]
