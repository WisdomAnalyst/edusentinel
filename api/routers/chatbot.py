"""Chatbot API — REST endpoint + WhatsApp webhook for EduBot."""

import uuid
from fastapi import APIRouter, Request, Response
from loguru import logger

from api.schemas.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

# Simple in-memory session store (use Redis in production)
_chains: dict[str, object] = {}


def _get_or_build_chain(session_id: str, language: str, grade: int, subject: str):
    key = f"{session_id}_{language}_{grade}_{subject}"
    if key not in _chains:
        from chatbot.rag.chain import build_chain
        _chains[key] = build_chain(
            language_code=language,
            grade_level=grade,
            subject=subject,
        )
    return _chains[key]


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        chain = _get_or_build_chain(session_id, req.language, req.grade_level, req.subject)
        result = chain.invoke({"question": req.message})
        answer = result.get("answer", str(result))
        sources = [
            doc.metadata.get("source", "NERDC Curriculum")
            for doc in result.get("source_documents", [])
        ]
    except Exception as e:
        logger.warning(f"Chain invoke failed ({e}) — using fallback")
        answer = (
            f"I'm here to help you learn {req.subject}! "
            f"Could you rephrase your question? I'm ready to assist."
        )
        sources = []

    return ChatResponse(
        answer=answer,
        language=req.language,
        sources=list(set(sources))[:3],
        session_id=session_id,
    )


@router.post("/whatsapp")
async def whatsapp_webhook(request: Request) -> Response:
    from chatbot.whatsapp.webhook import handle_whatsapp_message
    return await handle_whatsapp_message(request)


@router.get("/languages")
def list_languages():
    return [
        {"code": "en", "name": "English", "flag": "🇬🇧"},
        {"code": "ha", "name": "Hausa", "flag": "🇳🇬"},
        {"code": "yo", "name": "Yoruba", "flag": "🇳🇬"},
        {"code": "ig", "name": "Igbo", "flag": "🇳🇬"},
        {"code": "pcm", "name": "Nigerian Pidgin", "flag": "🇳🇬"},
    ]


@router.get("/subjects")
def list_subjects():
    return [
        "Mathematics", "English Language", "Basic Science",
        "Social Studies", "Civic Education",
        "Agricultural Science", "Health & Physical Education",
        "Home Economics", "Christian Religious Studies",
        "Islamic Religious Studies",
    ]
