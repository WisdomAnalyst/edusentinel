"""
Language detection and translation utilities.
Detects query language and translates between supported languages.
"""

import os
from loguru import logger

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "pcm": "Nigerian Pidgin",
}

# Lightweight keyword heuristics for language detection
LANGUAGE_MARKERS = {
    "ha": ["menene", "yaya", "ina", "taimaka", "koyar", "sannu", "don", "wane"],
    "yo": ["bawo", "jẹ", "kini", "sọ", "fun", "kọ", "ẹ", "àwọn", "nípa"],
    "ig": ["gịnị", "bụ", "ọ", "dị", "mma", "nnọọ", "mụta", "anyị"],
    "pcm": ["wetin", "how far", "abeg", "naim", "dey", "wey", "dem", "una"],
}


def detect_language(text: str) -> str:
    """Heuristic language detection for Nigeria's 5 supported languages."""
    text_lower = text.lower()
    scores = {lang: 0 for lang in LANGUAGE_MARKERS}
    for lang, markers in LANGUAGE_MARKERS.items():
        for marker in markers:
            if marker in text_lower:
                scores[lang] += 1
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] > 0:
        logger.debug(f"Detected language: {best_lang} (score: {scores[best_lang]})")
        return best_lang
    return "en"


def translate_with_gemini(text: str, target_lang: str) -> str:
    """Translate text using Gemini when needed."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return text
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        lang_name = SUPPORTED_LANGUAGES.get(target_lang, "English")
        prompt = (
            f"Translate the following educational text to {lang_name}. "
            f"Keep it simple and child-friendly.\n\nText: {text}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}, returning original text")
        return text
