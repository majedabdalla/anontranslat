"""
parser.py
---------
Parsing logic for the admin-group log message format:

    📢 Room #ac450470
    👤 Sender: ay | @hytnfusss (ID: 779442808, phone: )
    👥 Receiver: . | @titiktiktiktik (ID: 1109210465, phone: )
    Room Created: 178150243.19796

    💬 Message: <free text>

Only the text after "💬 Message:" is ever extracted or used. The
Room/Sender/Receiver/Created lines are only used to confirm that a
message actually has the expected log shape -- their contents (IDs,
usernames, phone numbers) are never parsed out or acted upon by this
module or anything downstream of it.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Compiled once at import time since this regex runs against every
# group message the bot sees.
#
# - re.DOTALL lets '.' match newlines too, so the lazy ".*?" sections
#   can span the multi-line header block between markers.
# - re.IGNORECASE adds tolerance for casing variants of "Room", "Sender",
#   "Receiver", "Created" without weakening the structural check (the
#   emoji markers themselves are case-less, so this is safe).
_LOG_PATTERN = re.compile(
    r"💬\s*Message:\s*(?P<message>.+)",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedLog:
    """A successfully parsed log entry.

    Only the extracted message text is exposed here -- intentionally,
    since that is the only part of the log this bot ever forwards
    anywhere (to Gemini, and back into the reply).
    """

    message: str


def parse_log_message(text: str) -> Optional[ParsedLog]:
    """Parse `text` as an admin-group log entry, if it matches the format.

    Args:
        text: The raw Telegram message text to check.

    Returns:
        A `ParsedLog` containing the trimmed text after "💬 Message:"
        when `text` matches the expected Room/Sender/Receiver/Message
        structure and the extracted text is non-empty.
        `None` if `text` doesn't match the format, is missing the
        "💬 Message:" field entirely, or the extracted message is blank.
    """
    if not text or "💬" not in text:
        # Cheap rejection before running the full regex. Most messages
        # in a busy group won't contain this marker at all, so this
        # avoids paying regex cost on every single message.
        return None

    match = _LOG_PATTERN.search(text)
    if not match:
        return None

    extracted = match.group("message").strip()
    if not extracted:
        return None

    return ParsedLog(message=extracted)
