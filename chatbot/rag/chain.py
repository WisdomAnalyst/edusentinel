"""
RAG Chain — builds the full conversational retrieval chain.
LangChain + Google Gemini + ChromaDB
Supports: English, Hausa, Yoruba, Igbo, Nigerian Pidgin
"""

import os
from pathlib import Path
from loguru import logger

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from chatbot.rag.ingestion import build_vector_store

LANGUAGE_PROMPTS = {
    "en": "Respond in clear, simple English appropriate for a primary school child.",
    "ha": "Amsa cikin Hausa mai sauƙi wanda yaron firamare zai iya fahimta.",
    "yo": "Dahun ni Yorùbá tí ó rọrùn tí ọmọ ilé-ìwé fípò kékeré le ye.",
    "ig": "Zaghachi n'Igbo dị mfe nke nwata ụlọ akwụkwọ nwere ike ghọta.",
    "pcm": "Answer in simple Nigerian Pidgin wey primary school pikin go understand.",
}

SYSTEM_TEMPLATE = """You are EduBot, an AI educational assistant for Kotlead —
a Nigerian NGO dedicated to education access for underserved children.

Your role: Help children learn Nigeria's NERDC primary school curriculum.

Guidelines:
- Use ONLY the curriculum context provided below to answer questions
- Keep explanations age-appropriate and simple
- Use local Nigerian examples (naira, local foods, famous places, Nigerian names)
- If you don't know something from the curriculum, say so honestly
- Always encourage the child and make learning fun
- {language_instruction}
- Grade level: Primary {grade_level}
- Subject: {subject}

CURRICULUM CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

STUDENT'S QUESTION: {question}

EduBot's Answer (in {language_name}, simple and encouraging):"""


def build_chain(
    language_code: str = "en",
    grade_level: int = 4,
    subject: str = "Mathematics",
) -> ConversationalRetrievalChain:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.4,
        max_output_tokens=1024,
        convert_system_message_to_human=True,
    )

    # Vector store & retriever
    vectorstore = build_vector_store()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "filter": {"subject": subject} if subject else None,
        },
    )

    lang_instruction = LANGUAGE_PROMPTS.get(language_code, LANGUAGE_PROMPTS["en"])
    lang_names = {
        "en": "English", "ha": "Hausa", "yo": "Yoruba",
        "ig": "Igbo", "pcm": "Nigerian Pidgin",
    }

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_TEMPLATE,
        partial_variables={
            "language_instruction": lang_instruction,
            "language_name": lang_names.get(language_code, "English"),
            "grade_level": str(grade_level),
            "subject": subject,
        },
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=8,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )

    logger.info(
        f"RAG chain built — language: {language_code}, "
        f"grade: {grade_level}, subject: {subject}"
    )
    return chain
