"""
EduSentinel FastAPI Backend
Kotlead | AI Platform for Education Access Intelligence

Endpoints:
  GET  /                      Health check + platform info
  GET  /dashboard/summary     National KPIs
  GET  /dashboard/lgas        LGA-level data (filterable)
  GET  /dashboard/hotspots    Top N critical LGAs
  POST /predictions/score-child  Individual dropout risk score
  POST /predictions/retrain   Trigger model retraining
  POST /chatbot/chat          EduBot REST API
  POST /chatbot/whatsapp      Twilio WhatsApp webhook
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import os

from api.routers import dashboard, predictions, chatbot


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EduSentinel API starting up …")
    # Pre-build ChromaDB vector store if Google API key is available
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from chatbot.rag.ingestion import build_vector_store
            build_vector_store()
            logger.success("ChromaDB vector store ready")
        except Exception as e:
            logger.warning(f"Vector store build skipped: {e}")
    yield
    logger.info("EduSentinel API shutting down")


app = FastAPI(
    title="EduSentinel API",
    description=(
        "AI Platform for Education Access Intelligence & Learning Support "
        "for Underserved Children in Nigeria. Built by Kotlead."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(predictions.router)
app.include_router(chatbot.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "platform": "EduSentinel",
        "organisation": "Kotlead",
        "version": "1.0.0",
        "status": "operational",
        "components": {
            "risk_intelligence_dashboard": "active",
            "dropout_early_warning": "active",
            "multilingual_ai_tutor": "active",
        },
        "coverage": "Nigeria — 36 states, 774 LGAs",
        "languages_supported": ["English", "Hausa", "Yoruba", "Igbo", "Nigerian Pidgin"],
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    from pathlib import Path
    model_dir = Path(__file__).parent.parent / "models" / "dropout_prediction" / "artifacts"
    chroma_dir = Path(__file__).parent.parent / "chroma_db"
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    return {
        "status": "healthy",
        "data_pipeline": "ready" if (data_dir / "nigeria_lga_education_indicators.csv").exists() else "needs_init",
        "ml_models": "ready" if (model_dir / "xgb_dropout.pkl").exists() else "not_trained",
        "vector_store": "ready" if chroma_dir.exists() else "not_built",
    }
