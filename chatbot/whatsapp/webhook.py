"""
WhatsApp Webhook — Twilio integration for EduBot on WhatsApp
Children can interact with EduBot via WhatsApp on basic phones.
Works with low-bandwidth connections.
Setup: https://www.twilio.com/docs/whatsapp/sandbox
"""

import os
import re
from fastapi import Request, Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from loguru import logger

from chatbot.languages.translator import detect_language
from chatbot.rag.chain import build_chain

# Per-user session state (in production: use Redis)
_sessions: dict[str, dict] = {}
_chains: dict[str, object] = {}

WELCOME_MSG = {
    "en": (
        "👋 Welcome to *EduBot* by Kotlead!\n\n"
        "I'm your multilingual learning assistant. I can help you learn Mathematics, "
        "English, Science, and more — in English, Hausa, Yoruba, Igbo, or Pidgin.\n\n"
        "Reply with:\n"
        "• *english* — English mode\n"
        "• *hausa* — Hausa mode\n"
        "• *yoruba* — Yoruba mode\n"
        "• *igbo* — Igbo mode\n"
        "• *pidgin* — Pidgin mode\n\n"
        "Or just ask me a question! 📚"
    ),
    "ha": "👋 Barka da zuwa *EduBot* na Kotlead!\nAmmana naka: Koyon ilimi cikin harshenka.",
    "yo": "👋 Ẹ káàbọ̀ sí *EduBot* ti Kotlead!\nOlùranlọ́wọ́ ẹ̀kọ́ rẹ.",
    "ig": "👋 Nnọọ na *EduBot* nke Kotlead!\nOnye enyemaka mụta gị.",
    "pcm": "👋 Welcome to *EduBot* by Kotlead!\nYour learning helper for WhatsApp.",
}

LANG_COMMANDS = {
    "english": "en", "hausa": "ha", "haoussa": "ha",
    "yoruba": "yo", "igbo": "ig", "ibo": "ig",
    "pidgin": "pcm", "pidgeon": "pcm",
}

GRADE_PATTERN = re.compile(r"\b(primary|pry|grade|jss)\s*(\d)\b", re.IGNORECASE)


async def handle_whatsapp_message(request: Request) -> Response:
    """Main webhook handler for incoming WhatsApp messages."""
    form = await request.form()
    body = str(form.get("Body", "")).strip()
    from_number = str(form.get("From", ""))
    to_number = str(form.get("To", ""))

    logger.info(f"WA message from {from_number}: {body[:80]}")

    # Validate Twilio signature in production
    if os.getenv("TWILIO_AUTH_TOKEN"):
        validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN"))
        url = str(request.url)
        signature = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(url, dict(form), signature):
            logger.warning("Invalid Twilio signature")
            return Response(status_code=403)

    resp = MessagingResponse()
    reply = await _process_message(from_number, body)
    resp.message(reply)
    return Response(content=str(resp), media_type="application/xml")


async def _process_message(user_id: str, message: str) -> str:
    session = _sessions.setdefault(user_id, {
        "language": "en",
        "grade": 4,
        "subject": "Mathematics",
        "message_count": 0,
    })
    session["message_count"] += 1

    msg_lower = message.lower().strip()

    # Command: language switch
    if msg_lower in LANG_COMMANDS:
        session["language"] = LANG_COMMANDS[msg_lower]
        lang = session["language"]
        _chains.pop(user_id, None)
        return f"✅ Language set to {msg_lower.title()}.\n" + WELCOME_MSG.get(lang, WELCOME_MSG["en"])

    # Command: help / start
    if msg_lower in ("hi", "hello", "start", "help", "menu"):
        lang = detect_language(message) if len(message) > 3 else session["language"]
        session["language"] = lang
        return WELCOME_MSG.get(lang, WELCOME_MSG["en"])

    # Command: grade level
    grade_match = GRADE_PATTERN.search(message)
    if grade_match:
        session["grade"] = int(grade_match.group(2))
        _chains.pop(user_id, None)
        return f"✅ Grade set to {session['grade']}. Ask me any question!"

    # Detect language from message
    detected = detect_language(message)
    if detected != "en":
        session["language"] = detected

    # Get or build chain
    chain_key = f"{user_id}_{session['language']}_{session['grade']}_{session['subject']}"
    if chain_key not in _chains:
        try:
            _chains[chain_key] = build_chain(
                language_code=session["language"],
                grade_level=session["grade"],
                subject=session["subject"],
            )
        except Exception as e:
            logger.error(f"Chain build failed: {e}")
            return (
                "⚠️ EduBot is starting up. Please try again in a moment.\n"
                "Reply *help* for options."
            )

    chain = _chains[chain_key]
    try:
        result = chain.invoke({"question": message})
        answer = result.get("answer", str(result))
        # Truncate for WhatsApp (1600 char limit)
        if len(answer) > 1500:
            answer = answer[:1497] + "…"
        return answer
    except Exception as e:
        logger.error(f"Chain invoke failed: {e}")
        return "I couldn't answer that. Please rephrase your question or reply *help*."
