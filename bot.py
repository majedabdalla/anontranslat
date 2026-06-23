"""
bot.py
------
Entry point and Telegram wiring.

Listens for text messages in the configured admin group, parses the
specific Room/Sender/Receiver/Message log format (see parser.py),
sends only the extracted "💬 Message:" text to Gemini for translation
(see gemini_client.py), and replies directly to the original log
message with the English translation.

Run with:  python bot.py
Stop with: Ctrl+C (python-telegram-bot shuts down polling cleanly).
"""

import logging
from typing import Optional

from telegram import Message, Update
from telegram.error import TelegramError
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from config import Settings, load_settings
from gemini_client import GeminiTranslator, TranslationError
from parser import parse_log_message

logger = logging.getLogger(__name__)


def configure_logging(level_name: str) -> None:
    """Set up clean, structured logging for the whole process.

    Third-party libraries (httpx, the telegram library itself) log a
    lot of per-request noise at INFO level -- this keeps that at
    WARNING so the logs stay focused on this bot's own decisions.
    """
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)


def _log_preview(text: str, limit: int = 60) -> str:
    """Collapse whitespace and truncate `text` for safe, compact logging.

    This is the only place extracted message content ever touches the
    logs, and only as a short preview -- never the full body, and
    never the sender/receiver/phone/ID fields from the original log.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main handler: runs for every text message the bot sees in a group.

    Cheaply ignores anything that isn't a genuine log message, then
    translates and replies only to confirmed matches.
    """
    message = update.effective_message
    if message is None or not message.text:
        return

    # Ignore the bot's own messages -- prevents any possibility of a
    # translation reply ever being picked back up and re-processed.
    if message.from_user and message.from_user.id == context.bot.id:
        return

    # Optional hard scoping to a single configured group, for safety
    # (e.g. if the bot is ever added to another chat by mistake).
    admin_group_id: Optional[int] = context.bot_data.get("admin_group_id")
    if admin_group_id is not None and message.chat_id != admin_group_id:
        return

    parsed = parse_log_message(message.text)
    if parsed is None:
        # Not a log message in the expected format -- this is the
        # normal case for most group chatter, so this stays silent.
        return

    logger.info(
        "Matched log message | chat_id=%s message_id=%s text=%r",
        message.chat_id,
        message.message_id,
        _log_preview(parsed.message),
    )

    translator: GeminiTranslator = context.bot_data["translator"]

    try:
        translation = await translator.translate(parsed.message)
    except TranslationError as exc:
        logger.warning(
            "Translation failed | chat_id=%s message_id=%s reason=%s",
            message.chat_id,
            message.message_id,
            exc,
        )
        await _safe_reply(context, message, "⚠️ Translation unavailable for this message.")
        return

    await _safe_reply(context, message, translation)


async def _safe_reply(context: ContextTypes.DEFAULT_TYPE, message: Message, text: str) -> None:
    """Reply to `message` with `text`, never letting a Telegram API
    error (missing permissions, deleted message, network blip, etc.)
    propagate up and crash the bot.
    """
    try:
        await context.bot.send_message(
            chat_id=message.chat_id,
            text=text,
            reply_to_message_id=message.message_id,
            # If the original message gets deleted between us parsing
            # it and sending the reply, Telegram would otherwise reject
            # the whole reply -- this makes it fall back to a plain
            # message instead of failing outright.
            allow_sending_without_reply=True,
        )
    except TelegramError as exc:
        logger.error(
            "Failed to send reply | chat_id=%s message_id=%s error=%s",
            message.chat_id,
            message.message_id,
            exc,
        )


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler so any unexpected exception in a handler is
    logged instead of silently killing the polling loop."""
    logger.error("Unhandled exception while processing an update", exc_info=context.error)


def build_application(settings: Settings) -> Application:
    """Construct and wire up the PTB Application from validated settings."""
    application = Application.builder().token(settings.telegram_bot_token).build()

    # Shared, long-lived objects live in bot_data instead of module-level
    # globals -- keeps handlers testable and avoids hidden state.
    application.bot_data["translator"] = GeminiTranslator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    application.bot_data["admin_group_id"] = settings.admin_group_id

    # Only plain text messages in groups/supergroups, excluding commands --
    # everything this bot cares about is plain chat text, never a command.
    application.add_handler(
        MessageHandler(filters.ChatType.GROUPS & filters.TEXT & ~filters.COMMAND, handle_group_message)
    )
    application.add_error_handler(handle_error)

    return application


def main() -> None:
    """Load configuration, build the bot, and run it until interrupted."""
    settings = load_settings()
    configure_logging(settings.log_level)

    logger.info(
        "Starting bot | model=%s admin_group_id=%s",
        settings.gemini_model,
        settings.admin_group_id if settings.admin_group_id is not None else "ANY",
    )

    application = build_application(settings)

    # run_polling() blocks here, managing its own event loop and
    # shutting down cleanly on Ctrl+C / SIGTERM.
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
