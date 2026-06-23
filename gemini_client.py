"""
gemini_client.py
-----------------
Async Gemini translation client.

Uses Google's current unified Gen AI SDK, `google-genai`
(imported as `from google import genai`) -- this is the actively
maintained SDK, distinct from the older, now-legacy
`google-generativeai` package. It exposes a native async interface
via `client.aio`, which is what lets every Gemini call in this file
run without ever blocking the bot's asyncio event loop.

Switching models later (e.g. once a newer or cheaper free-tier model
ships) only requires changing the GEMINI_MODEL environment variable --
nothing in this file needs to change.
"""

import asyncio
import logging
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Strict system instruction for the translation model. This is the only
# "personality" the model is given -- it should never explain, hedge,
# label, or comment, only translate.
_SYSTEM_INSTRUCTION = (
    "You are a translation engine specialized in informal Indonesian "
    "internet slang, abbreviated text speak, and casual chat language.\n\n"
    "Translate the user's message into fluent, natural English.\n\n"
    "Rules:\n"
    "- Infer the intended meaning even when the source is fragmented, "
    "has typos, repeats words, or omits the subject.\n"
    "- Preserve the original tone (e.g. rude, casual, sarcastic, "
    "affectionate, angry) wherever possible -- this is a meaning-for-"
    "meaning translation, not a literal word-for-word one.\n"
    "- Output ONLY the translated text. No labels like 'Translation:', "
    "no notes, no explanations, no analysis, no disclaimers.\n"
    "- Do not wrap the output in quotation marks.\n"
    "- Do not use markdown or any other formatting unless the source "
    "text itself requires it.\n"
    "- If the source text is already in English, return a cleaned-up "
    "version with the same meaning."
)

# Generation is deliberately low-temperature and short -- this is a
# translation task, not a creative one, and source messages are short
# chat lines.
_TEMPERATURE = 0.3
_MAX_OUTPUT_TOKENS = 512


class TranslationError(Exception):
    """Raised whenever a translation could not be produced.

    Callers (the Telegram handler) catch this single exception type
    instead of needing to know about Gemini SDK internals, timeouts,
    or network errors individually.
    """


class GeminiTranslator:
    """Thin async wrapper around the Gemini API for one job only:
    turning informal Indonesian chat text into natural English.
    """

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 20.0) -> None:
        # genai.Client talks to the Gemini Developer API (the free-tier,
        # API-key based endpoint) by default -- no Vertex AI / GCP
        # project setup required.
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def translate(self, text: str) -> str:
        """Translate `text` (informal Indonesian) into natural English.

        Args:
            text: Raw extracted message content to translate.

        Returns:
            The translated English text, stripped of surrounding
            whitespace.

        Raises:
            TranslationError: on empty input, request timeout, any
            Gemini API failure, or an empty/unusable response. Callers
            decide how to react (e.g. reply with a short error, or stay
            silent) without needing to know why it failed.
        """
        cleaned = text.strip()
        if not cleaned:
            raise TranslationError("Cannot translate empty text.")

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents=cleaned,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_INSTRUCTION,
                        temperature=_TEMPERATURE,
                        max_output_tokens=_MAX_OUTPUT_TOKENS,
                    ),
                ),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("Gemini request timed out after %.1fs", self._timeout_seconds)
            raise TranslationError("Translation request timed out.") from exc
        except Exception as exc:
            # Covers Gemini SDK errors (auth, rate limiting, server
            # errors) and transport-level network failures alike. The
            # exact exception type isn't pattern-matched here so this
            # stays correct even as the SDK's error hierarchy evolves
            # across versions.
            logger.error("Gemini API call failed: %s", exc)
            raise TranslationError("Translation request failed.") from exc

        translated = _extract_text(response)
        if not translated:
            logger.warning("Gemini returned an empty or unusable response.")
            raise TranslationError("Translation returned no usable text.")

        return translated


def _extract_text(response: object) -> Optional[str]:
    """Defensively pull plain text out of a Gemini response.

    `response.text` is the normal path. It can be missing -- e.g. if
    the response was blocked by a safety filter and has no candidates --
    so this avoids letting an AttributeError escape as an unhandled
    crash.
    """
    text = getattr(response, "text", None)
    return text.strip() if text else None
