"""
Page 3 - Multilingual AI Learning Assistant
RAG-powered educational chatbot grounded in Nigeria's NERDC curriculum.
Supports English, Hausa, Yoruba, Igbo, and Nigerian Pidgin.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="AI Tutor | EduSentinel", layout="wide", page_icon="=?")

LANGUAGES = {
    "English":         {"code": "en", "flag": "EN", "greeting": "Hello! I'm EduBot, your learning assistant. What would you like to learn today?"},
    "Hausa":           {"code": "ha", "flag": "HA", "greeting": "Sannu! Ni ne EduBot, mataimakin koyo naka. Menene kake so ka koyi yau?"},
    "Yoruba":          {"code": "yo", "flag": "YO", "greeting": "Bawo! Mo je EduBot, oluranlowo ikeko re. Kini o fe ko loni?"},
    "Igbo":            {"code": "ig", "flag": "IG", "greeting": "Nnoo! Abu m EduBot, onye enyemaka muta gi. Gini choroo i muta taa?"},
    "Nigerian Pidgin": {"code": "pcm", "flag": "PID", "greeting": "How far! Na me be EduBot, your learning helper. Wetin you wan learn today?"},
}

SUBJECTS = {
    "Mathematics": "Math",
    "English Language": "English",
    "Basic Science": "Science",
    "Social Studies": "Social",
    "Civic Education": "Civic",
    "Agricultural Science": "Agric",
    "Health & Physical Education": "Health",
    "Home Economics": "Home Econ",
}

GRADE_LEVELS = {
    "Primary 1 (Age 6-7)": 1,
    "Primary 2 (Age 7-8)": 2,
    "Primary 3 (Age 8-9)": 3,
    "Primary 4 (Age 9-10)": 4,
    "Primary 5 (Age 10-11)": 5,
    "Primary 6 (Age 11-12)": 6,
    "JSS 1 (Age 12-13)": 7,
    "JSS 2 (Age 13-14)": 8,
    "JSS 3 (Age 14-15)": 9,
}

# ── Demo response function defined FIRST ─────────────────────────────────────
def _demo_response(question: str, lang: str, subject: str, grade: int) -> str:
    q_lower = question.lower()

    DEMO_ANSWERS = {
        "en": {
            "addition": (
                "Great question! Addition means putting numbers together.\n\n"
                "**Example:** 5 + 3 = 8\n\n"
                "Imagine you have **5 mangoes** and your friend gives you **3 more**. "
                "Now you have **8 mangoes** total!\n\n"
                "**Practice:** What is 7 + 4? Try it!"
            ),
            "fraction": (
                "A **fraction** shows part of a whole thing.\n\n"
                "If a pizza is cut into **4 equal slices** and you eat **1 slice**, "
                "you ate **1/4** (one-quarter) of the pizza.\n\n"
                "- The **bottom number** (denominator) = total parts\n"
                "- The **top number** (numerator) = parts you have\n\n"
                "**Example:** 3/4 means 3 out of 4 parts."
            ),
            "noun": (
                "A **noun** is a word that names a person, place, animal, or thing.\n\n"
                "**Examples:**\n"
                "- Person: *teacher, child, doctor*\n"
                "- Place: *Lagos, school, market*\n"
                "- Animal: *dog, goat, eagle*\n"
                "- Thing: *book, pencil, water*\n\n"
                "**Sentence:** *The teacher gave the child a book.*\n"
                "(teacher, child, book - all nouns!)"
            ),
            "water cycle": (
                "The **water cycle** is how water moves around the Earth!\n\n"
                "**4 steps:**\n"
                "1. **Evaporation** - Sun heats water in rivers and oceans, turning it into invisible water vapour that rises up\n"
                "2. **Condensation** - High in the sky, vapour cools and forms clouds\n"
                "3. **Precipitation** - Water falls as rain or snow\n"
                "4. **Collection** - Water collects in rivers and lakes, then the cycle starts again\n\n"
                "This is why there is always water on Earth!"
            ),
            "multiplication": (
                "**Multiplication** is fast addition!\n\n"
                "3 x 4 means: 3 + 3 + 3 + 3 = 12\n\n"
                "**Times Tables tip:**\n"
                "- 2 x 5 = 10\n"
                "- 3 x 5 = 15\n"
                "- 4 x 5 = 20\n\n"
                "**Real life:** If 1 bag of rice costs N500, then 3 bags cost N500 x 3 = **N1,500**"
            ),
            "default": (
                f"Hello! I am here to help you learn **{subject}** for Grade {grade}.\n\n"
                f"Ask me to:\n"
                f"- Explain a topic (e.g. 'What is addition?')\n"
                f"- Give you practice questions\n"
                f"- Tell you a story that teaches a lesson\n\n"
                f"I am ready to help! What would you like to learn?"
            ),
        },
        "ha": {
            "default": (
                f"Sannu! Ina nan don taimaka maka ka koyi **{subject}**.\n\n"
                f"Kana iya tambayata tambayoyi game da:\n"
                f"- Bayanin zango\n"
                f"- Tambayoyin aiki\n"
                f"- Labarai masu koyarwa\n\n"
                f"Bari mu fara koyo tare!"
            ),
        },
        "yo": {
            "default": (
                f"E kaabo! Mo wa nibi lati ran o lowo lati ko **{subject}**.\n\n"
                f"O le beere lowo mi nipa:\n"
                f"- Itumo eko\n"
                f"- Awon idanwo\n"
                f"- Itan awon eko\n\n"
                f"Je ka ko papo!"
            ),
        },
        "ig": {
            "default": (
                f"Nnoo! Abu m ebe a inyere gi aka imuta **{subject}**.\n\n"
                f"I nwere ike juoo m ajujuo gbasara:\n"
                f"- Nkowa ihe omumu\n"
                f"- Ajujuo omumu\n"
                f"- Akuko ihe mere eme\n\n"
                f"Ka anyi muta onu!"
            ),
        },
        "pcm": {
            "default": (
                f"How far! I dey here to help you learn **{subject}**.\n\n"
                f"You fit ask me:\n"
                f"- Make I explain any topic\n"
                f"- Give you practice question\n"
                f"- Tell you story wey go teach you\n\n"
                f"Let us learn together!"
            ),
        },
    }

    lang_code_map = {
        "English": "en", "Hausa": "ha",
        "Yoruba": "yo", "Igbo": "ig", "Nigerian Pidgin": "pcm"
    }
    lc = lang_code_map.get(lang, "en")
    lang_answers = DEMO_ANSWERS.get(lc, DEMO_ANSWERS["en"])

    for keyword, answer in lang_answers.items():
        if keyword != "default" and keyword in q_lower:
            return answer
    return lang_answers.get("default", DEMO_ANSWERS["en"]["default"])


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.chat-user {
    background: #1e3a5f; border-radius:12px 12px 4px 12px;
    padding:0.8rem 1rem; margin:0.5rem 0; max-width:80%; margin-left:auto;
    color:#e8edf5; border-left: 3px solid #3b82f6;
}
.chat-bot {
    background: #0d2137; border-radius:12px 12px 12px 4px;
    padding:0.8rem 1rem; margin:0.5rem 0; max-width:85%;
    color:#e8edf5; border-left: 3px solid #10b981;
}
.chat-source {
    background:#071523; border-radius:6px; padding:0.4rem 0.7rem;
    margin-top:0.4rem; font-size:0.8rem; color:#64748b;
    border-left: 2px solid #334155;
}
</style>
""", unsafe_allow_html=True)

st.title("EduBot - Multilingual AI Learning Assistant")
st.markdown(
    "Powered by **Google Gemini** + **RAG** over Nigeria's **NERDC curriculum**. "
    "Supports 5 languages. Works on low-bandwidth connections."
)

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Tutor Settings")
    selected_lang = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        format_func=lambda x: f"[{LANGUAGES[x]['flag']}] {x}",
    )
    selected_grade = st.selectbox("Grade Level", list(GRADE_LEVELS.keys()))
    selected_subject = st.selectbox(
        "Subject",
        list(SUBJECTS.keys()),
        format_func=lambda x: f"{SUBJECTS[x]} - {x}",
    )
    st.markdown("---")
    low_bandwidth = st.toggle("Low-bandwidth mode", value=True)
    show_sources = st.toggle("Show curriculum sources", value=True)
    st.markdown("---")
    st.markdown("**Supported Languages:**")
    for lang in LANGUAGES:
        st.markdown(f"- {lang}")
    st.markdown("---")
    st.caption(
        "Grounded in the NERDC National Curriculum (Basic Education). "
        "All answers cited to curriculum source."
    )

# ── RAG chain loader ──────────────────────────────────────────────────────────
@st.cache_resource
def load_rag_chain(language_code: str, grade: int, subject: str):
    try:
        from chatbot.rag.chain import build_chain
        chain = build_chain(language_code=language_code, grade_level=grade, subject=subject)
        return chain, True
    except Exception as e:
        return None, False

lang_code = LANGUAGES[selected_lang]["code"]
grade_num = GRADE_LEVELS[selected_grade]
chain, chain_loaded = load_rag_chain(lang_code, grade_num, selected_subject)

# ── Session init ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("active_lang") != selected_lang:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": LANGUAGES[selected_lang]["greeting"],
            "sources": [],
        }
    ]
    st.session_state["active_lang"] = selected_lang

# ── Context bar ───────────────────────────────────────────────────────────────
ctx1, ctx2, ctx3 = st.columns(3)
ctx1.info(f"**Language:** {selected_lang}")
ctx2.info(f"**Grade:** {selected_grade}")
ctx3.info(f"**Subject:** {selected_subject}")

# ── Chat display ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>You: {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>EduBot: {msg['content']}</div>", unsafe_allow_html=True)
        if show_sources and msg.get("sources"):
            for src in msg["sources"]:
                st.markdown(
                    f"<div class='chat-source'>Source: {src}</div>",
                    unsafe_allow_html=True,
                )

# ── Quick-start prompts ───────────────────────────────────────────────────────
if len(st.session_state.messages) == 1:
    st.markdown("**Try asking:**")
    QUICK_PROMPTS = {
        "Mathematics": [
            "Explain addition with examples",
            "What are fractions?",
            "Teach me multiplication tables",
        ],
        "English Language": [
            "What is a noun?",
            "How do I write a letter?",
            "What are vowels and consonants?",
        ],
        "Basic Science": [
            "What are the states of matter?",
            "Explain the water cycle",
            "Why do plants need sunlight?",
        ],
    }
    prompts = QUICK_PROMPTS.get(selected_subject, [
        "Teach me something new",
        "Give me a practice question",
        "Explain this subject to me",
    ])
    cols = st.columns(len(prompts))
    for i, (col, p) in enumerate(zip(cols, prompts)):
        if col.button(p, key=f"quick_{i}"):
            st.session_state.messages.append({"role": "user", "content": p, "sources": []})
            st.rerun()

# ── Input box ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    f"Ask EduBot in {selected_lang} ({selected_subject} - {selected_grade})"
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})

    with st.spinner("EduBot is thinking..."):
        if chain_loaded and chain is not None:
            try:
                result = chain.invoke({
                    "question": user_input,
                    "language": selected_lang,
                    "grade": grade_num,
                    "subject": selected_subject,
                })
                answer = result.get("answer", str(result))
                sources = [
                    doc.metadata.get("source", "NERDC Curriculum")
                    for doc in result.get("source_documents", [])
                ]
            except Exception as e:
                answer = f"I encountered an issue. Please try rephrasing your question."
                sources = []
        else:
            answer = _demo_response(user_input, selected_lang, selected_subject, grade_num)
            sources = [f"NERDC National Curriculum - {selected_subject}, Grade {grade_num}"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources if show_sources else [],
    })
    st.rerun()

# ── Bottom stats ──────────────────────────────────────────────────────────────
st.markdown("---")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Messages", len(st.session_state.messages))
s2.metric("Language", selected_lang)
s3.metric("Curriculum", "NERDC 2023")
s4.metric("RAG Engine", "Gemini + ChromaDB" if chain_loaded else "Demo Mode")
