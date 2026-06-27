"""
RAG Ingestion — embeds NERDC curriculum chunks into ChromaDB vector store.
Run once (or when curriculum updates) to build the knowledge base.
"""

from pathlib import Path
import os
from loguru import logger

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chatbot.curriculum.nerdc_loader import load_all_documents

CHROMA_DIR = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION_NAME = "nerdc_curriculum"


def get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """Embed curriculum docs and persist to ChromaDB."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if not force_rebuild:
        count = vectorstore._collection.count()
        if count > 0:
            logger.info(f"ChromaDB already has {count} chunks — skipping rebuild")
            return vectorstore

    logger.info("Building ChromaDB vector store from NERDC curriculum …")
    docs = load_all_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks from {len(docs)} source documents")

    vectorstore.add_documents(chunks)
    logger.success(f"ChromaDB built with {len(chunks)} chunks — persisted to {CHROMA_DIR}")
    return vectorstore


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    build_vector_store(force_rebuild=True)
