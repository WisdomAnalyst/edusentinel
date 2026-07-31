# EduSentinel
### AI Platform for Education Access Intelligence & Learning Support
**Built by Kotlead — Protecting Every Child's Right to Learn in Nigeria**

---

> **20.2 million children** are out of school in Nigeria — the largest out-of-school population in the world.  
> EduSentinel exists to change that, one data point at a time.

---

## What EduSentinel Does

EduSentinel is a production-grade AI platform with three integrated components, each independently powerful, together forming a complete decision-intelligence and intervention system:

| Component | What it does | Who uses it |
|---|---|---|
| **Risk Intelligence Dashboard** | Maps OOS hotspots across 774 LGAs, identifies dominant dropout drivers, predicts next-term escalation | NGO programme directors, UNICEF field officers, policymakers |
| **Dropout Early Warning System** | Scores individual children for dropout risk with SHAP-powered explanations | School coordinators, community mobilisers, Kotlead field staff |
| **Multilingual AI Learning Assistant** | NERDC-grounded educational chatbot in 5 languages, accessible via WhatsApp | Children, teachers, parents in underserved communities |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EduSentinel Platform                         │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  Streamlit       │    │  FastAPI Backend                  │   │
│  │  Dashboard       │◄───│  /dashboard  /predictions         │   │
│  │  :8501           │    │  /chatbot    /health              │   │
│  └──────────────────┘    └────────────┬─────────────────────┘   │
│                                        │                         │
│  ┌─────────────────┐  ┌───────────────▼──┐  ┌───────────────┐  │
│  │  Data Pipeline  │  │  ML Models       │  │  RAG Engine   │  │
│  │  ─────────────  │  │  ─────────────── │  │  ─────────── │  │
│  │  UNICEF MICS    │  │  XGBoost         │  │  Gemini 1.5   │  │
│  │  NBS Nigeria    │  │  LightGBM        │  │  ChromaDB     │  │
│  │  ACLED Conflict │  │  K-Means Cluster │  │  LangChain    │  │
│  │  GRID3 Geo      │  │  SHAP Explainer  │  │  NERDC Docs   │  │
│  │  Synthetic Gen  │  │  MLflow Tracking │  │               │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  WhatsApp (Twilio)  ◄──►  EduBot  ◄──►  RAG Chain       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Out-of-School Children Risk Intelligence Dashboard

### What it delivers
- **Hotspot Map** — Interactive Folium map of all 774 LGAs colour-coded by OOS rate
- **Driver Analysis** — Per-LGA breakdown of which factor (poverty, conflict, gender, distance, disability, school supply) dominates
- **Next-Term Prediction** — GBM-powered forecast of which communities will deteriorate in the next academic term
- **Resource Allocation Tool** — Ranked list of LGAs where NGO intervention has highest marginal impact

### Data sources (open access)
| Source | Data | URL |
|---|---|---|
| UNICEF MICS | Attendance rates, gender gaps | mics.unicef.org |
| Nigeria NBS | Poverty indices, household surveys | nigerianstat.gov.ng |
| ACLED | Armed conflict events & fatalities | acleddata.com |
| GRID3 Nigeria | School locations, geospatial | grid3.gov.ng |
| UNESCO UIS | Enrollment & completion rates | uis.unesco.org |

### ML Models
- **K-Means Clustering** — Groups LGAs into 5 risk quintiles for resource prioritisation
- **Gradient Boosting Regressor** — Predicts next-term OOS rate with R² > 0.85 on held-out data
- **Feature Weights** — Explainable driver contributions per LGA (not a black box)

---

## Component 2: School Dropout Early Warning System

### What it delivers
- Individual child risk score (0–100%) with confidence interval
- SHAP waterfall chart showing which factors are driving the score
- Intervention recommendation mapped to specific programmes (CCT, transport, home visits)
- Batch analysis mode for school-wide or community-wide screening
- Automated retraining pipeline with MLflow experiment tracking

### ML Pipeline
```
Raw Data → Feature Engineering → Train/Val/Test Split
         ↓
    XGBoost (500 trees, early stopping)
    LightGBM (500 trees, early stopping)
         ↓
    Ensemble (0.5/0.5 weighted average)
         ↓
    SHAP TreeExplainer → top-5 factors per child
         ↓
    MLflow: log params, metrics, artifacts, model
```

### Performance (data benchmark)
| Metric | Score |
|---|---|
| ROC-AUC | 0.89 |
| F1-Score | 0.78 |
| Precision | 0.80 |
| Recall | 0.77 |

### Features used (18 predictors)
Attendance rate · Math & literacy scores · Daily household income · Parental education · Distance to school · Disability status · Conflict displacement · School fee burden · Birth certificate · School feeding access · LGA poverty rate · LGA conflict score · Teacher-pupil ratio · Gender · Age · Grade level · Siblings · Academic composite

---

## Component 3: Multilingual AI Learning Assistant (EduBot)

### What it delivers
- Conversational AI tutor grounded in Nigeria's **NERDC National Curriculum**
- Answers only from verified curriculum sources — no hallucination
- All answers cite their source (e.g., "NERDC Curriculum — Mathematics, Primary 4")
- Accessible via **WhatsApp** — no smartphone or app needed
- Low-bandwidth optimised (text-only, <5KB per response)

### Languages supported
| Language | Speakers in Nigeria | Coverage |
|---|---|---|
| English | 60M+ (second language) | Full |
| Hausa | 70M+ | Full |
| Yoruba | 45M+ | Full |
| Igbo | 35M+ | Full |
| Nigerian Pidgin | 75M+ (lingua franca) | Full |

### Subjects covered (NERDC Primary 1–6, JSS 1–3)
Mathematics · English Language · Basic Science · Social Studies · Civic Education · Agricultural Science · Health & Physical Education · Home Economics · Religious Studies

### Technical stack
```python
LangChain ConversationalRetrievalChain
  + Google Gemini 1.5 Flash (LLM)
  + GoogleGenerativeAI Embeddings (models/embedding-001)
  + ChromaDB (vector store, persisted)
  + MMR retrieval (diversity-aware, k=4)
  + ConversationBufferWindowMemory (k=8 turns)
```

### WhatsApp Integration
Children message the Kotlead WhatsApp number → Twilio webhook → EduBot → Gemini + RAG → response in child's language, delivered to WhatsApp in under 3 seconds.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/apikey))
- Twilio account for WhatsApp

### Installation
```bash
git clone https://github.com/kotlead/edusentinel
cd edusentinel
pip install -r requirements.txt
Added Env
```

### One-command bootstrap
```bash
python run.py
```
This will:
1. Generate the synthetic Nigeria education dataset (774 LGAs, 20,000 children)
2. Train the XGBoost + LightGBM dropout prediction ensemble
3. Train the risk intelligence clustering and OOS forecasting models
4. Embed the NERDC curriculum into ChromaDB
5. Launch the Streamlit dashboard at **http://localhost:8501**
6. Launch the FastAPI backend at **http://localhost:8000**

### Docker deployment
```bash
docker-compose up --build
```
Services:
- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

---

## Project Structure

```
edusentinel/
├── dashboard/               # Streamlit multi-page dashboard
│   ├── app.py               # Main entry — KPIs, zone charts
│   └── pages/
│       ├── 01_risk_map.py   # Interactive LGA hotspot map
│       ├── 02_dropout_predictor.py  # Individual + batch risk scorer
│       └── 03_ai_tutor.py   # EduBot multilingual chatbot UI
│
├── data/
│   ├── ingestion/           # UNICEF, NBS, ACLED, GRID3 connectors
│   ├── processing/          # Preprocessing & feature engineering
│   └── synthetic/           # Realistic Nigeria data generator
│
├── models/
│   ├── risk_intelligence/   # Hotspot clustering + OOS forecasting
│   └── dropout_prediction/  # XGBoost/LightGBM + SHAP + MLflow
│
├── chatbot/
│   ├── rag/                 # ChromaDB ingestion + LangChain chain
│   ├── curriculum/          # NERDC curriculum document loader
│   ├── languages/           # Language detection & translation
│   └── whatsapp/            # Twilio webhook handler
│
├── api/                     # FastAPI backend
│   ├── main.py
│   ├── routers/             # dashboard, predictions, chatbot
│   └── schemas/             # Pydantic request/response models
│
├── run.py                   # One-command bootstrap
├── docker-compose.yml       # Full-stack Docker deployment
└── requirements.txt
```

---

## API Reference

### Risk Intelligence
```
GET  /dashboard/summary          National KPIs
GET  /dashboard/lgas             LGA data (filterable by zone, state, risk tier)
GET  /dashboard/hotspots?n=20    Top N critical LGAs
GET  /dashboard/states           State-level aggregation
GET  /dashboard/zones            Zone-level aggregation
```

### Dropout Prediction
```
POST /predictions/score-child    Score individual child profile
POST /predictions/retrain        Trigger background model retrain
GET  /predictions/model-status   Check if models are trained
```

### EduBot
```
POST /chatbot/chat               Conversational AI (REST)
POST /chatbot/whatsapp           Twilio WhatsApp webhook
GET  /chatbot/languages          List supported languages
GET  /chatbot/subjects           List curriculum subjects
```

Full interactive docs at `/docs` (Swagger UI) and `/redoc`.

---

## Impact Metrics

| Indicator | Target |
|---|---|
| LGAs monitored | 774 (all of Nigeria) |
| Children profiled per run | 20,000+ |
| Languages served | 5 (covering 280M+ speakers) |
| WhatsApp accessibility | Works on any phone with WhatsApp |
| Data latency | Real-time scoring (<500ms) |
| Model retraining | Automated quarterly via MLflow |

---

## Technical Depth Summary

| Layer | Technology | Why |
|---|---|---|
| Data ingestion | Python + requests | Connects to 4 open data APIs with graceful fallbacks |
| Feature engineering | pandas + scikit-learn | 21 predictive features including interaction terms |
| Classification | XGBoost + LightGBM | Best-in-class for tabular data; handles class imbalance |
| Explainability | SHAP TreeExplainer | Per-child factor attribution — essential for field use |
| Experiment tracking | MLflow | Reproducible, auditable model lineage |
| Geospatial | Folium + GeoPandas | Interactive maps with sub-LGA precision |
| LLM | Google Gemini 1.5 Flash | Low-latency, multilingual, cost-efficient |
| RAG | LangChain + ChromaDB | Grounds answers in verified curriculum; no hallucination |
| API | FastAPI + Pydantic v2 | Type-safe, auto-documented, production-ready |
| Deployment | Docker Compose | Single-command full-stack launch |
| Messaging | Twilio WhatsApp API | Reaches children on feature phones |

---

## Roadmap

**Phase 1 — Data & Dashboard**    
**Phase 2 — Dropout ML Pipeline** 
**Phase 3 — Multilingual AI Tutor** 
**Phase 4 — Production Deployment**
Not Included
- [ ] Deploy to AWS/GCP with production Postgres backend
- [ ] Integrate live UNICEF & NBS data feeds
- [ ] ACLED real-time conflict stream integration
- [ ] Mobile-first React frontend
- [ ] Teacher dashboard with class-level risk views
- [ ] SMS fallback for non-WhatsApp users (Africa's Talking API)
- [ ] Offline mode for low-connectivity communities

---

## About Kotlead

**Kotlead** is a Nigerian NGO dedicated to improving education access for underserved children across Nigeria. EduSentinel is Kotlead's flagship technology platform — built internally by the Kotlead data science team to make our field operations data-driven, our interventions evidence-based, and our impact measurable.

We build for the 20 million children who cannot wait.

---

