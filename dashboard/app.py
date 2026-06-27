"""
EduSentinel Dashboard — Main Entry Point
Kotlead | AI Platform for Education Access Intelligence
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="EduSentinel | Kotlead",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {background: #0a1628;}
    [data-testid="stSidebar"] * {color: #e8edf5 !important;}
    .metric-card {
        background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 100%);
        border: 1px solid #2563eb;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.3rem;
        text-align: center;
    }
    .metric-card h2 {color: #60a5fa; font-size: 2.2rem; margin: 0;}
    .metric-card p {color: #94a3b8; font-size: 0.85rem; margin: 0.2rem 0 0;}
    .risk-critical {color: #ef4444; font-weight: bold;}
    .risk-high {color: #f97316; font-weight: bold;}
    .risk-medium {color: #eab308; font-weight: bold;}
    .risk-low {color: #22c55e; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {background: #0d2137; border-radius: 8px;}
    .stTabs [data-baseweb="tab"] {color: #94a3b8;}
    .stTabs [aria-selected="true"] {color: #60a5fa !important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='background:#1e3a5f;border-radius:8px;padding:0.5rem 1rem;"
        "text-align:center;font-weight:bold;color:#60a5fa;font-size:1.1rem;'>"
        "EduSentinel | Kotlead</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/01_risk_map.py", label="Risk Intelligence Map", icon="🗺️")
    st.page_link("pages/02_dropout_predictor.py", label="Dropout Early Warning", icon="⚠️")
    st.page_link("pages/03_ai_tutor.py", label="AI Learning Assistant", icon="🤖")
    st.markdown("---")
    st.markdown(
        "<small>Built by **Kotlead** | Powered by AI</small>",
        unsafe_allow_html=True,
    )

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
    <h1 style="font-size:3rem; color:#60a5fa; margin-bottom:0.2rem;">📚 EduSentinel</h1>
    <h3 style="color:#94a3b8; font-weight:400;">AI Platform for Education Access Intelligence</h3>
    <p style="color:#64748b;">Kotlead | Protecting Every Child's Right to Learn in Nigeria</p>
</div>
""", unsafe_allow_html=True)

# ── Load summary data ─────────────────────────────────────────────────────────
import json, pandas as pd
from pathlib import Path

@st.cache_data
def load_summary():
    p = Path(__file__).parent.parent / "data" / "raw" / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "total_oos_children": 18_500_000,
        "national_oos_rate": 0.37,
        "critical_lgas": 87,
        "high_risk_lgas": 142,
        "total_lgas": 774,
        "total_states": 37,
        "dropout_rate": 0.31,
    }

@st.cache_data
def load_lga_data():
    p = Path(__file__).parent.parent / "data" / "raw" / "nigeria_lga_education_indicators.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

summary = load_summary()
lga_df = load_lga_data()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
        <h2>{summary.get('total_oos_children', 0):,.0f}</h2>
        <p>Out-of-School Children</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <h2>{summary.get('national_oos_rate', 0):.0%}</h2>
        <p>National OOS Rate</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <h2 class="risk-critical">{summary.get('critical_lgas', 0)}</h2>
        <p>Critical-Risk LGAs</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <h2 class="risk-high">{summary.get('high_risk_lgas', 0)}</h2>
        <p>High-Risk LGAs</p></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
        <h2>{summary.get('dropout_rate', 0):.0%}</h2>
        <p>Predicted Dropout Rate</p></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Zone-level bar chart ──────────────────────────────────────────────────────
if not lga_df.empty:
    import plotly.express as px

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OOS Rate by Geopolitical Zone")
        zone_agg = lga_df.groupby("geopolitical_zone").agg(
            oos_rate=("oos_rate", "mean"),
            oos_count=("oos_count", "sum"),
        ).reset_index().sort_values("oos_rate", ascending=False)

        fig = px.bar(
            zone_agg,
            x="geopolitical_zone",
            y="oos_rate",
            color="oos_rate",
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
            labels={"geopolitical_zone": "Zone", "oos_rate": "OOS Rate"},
            text=zone_agg["oos_rate"].map(lambda x: f"{x:.0%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
            font_color="#e8edf5", showlegend=False,
            coloraxis_showscale=False, height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Dominant Risk Drivers Across Nigeria")
        driver_counts = lga_df["dominant_driver"].value_counts().reset_index()
        driver_counts.columns = ["driver", "lga_count"]
        fig2 = px.pie(
            driver_counts,
            names="driver",
            values="lga_count",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.45,
        )
        fig2.update_layout(
            plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
            font_color="#e8edf5", height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Top 10 critical states
    st.subheader("Top 10 States by OOS Children Count")
    state_agg = lga_df.groupby("state").agg(
        oos_count=("oos_count", "sum"),
        oos_rate=("oos_rate", "mean"),
    ).reset_index().nlargest(10, "oos_count")

    fig3 = px.bar(
        state_agg,
        x="oos_count",
        y="state",
        orientation="h",
        color="oos_rate",
        color_continuous_scale=["#22c55e", "#ef4444"],
        labels={"oos_count": "OOS Children", "state": "State", "oos_rate": "OOS Rate"},
    )
    fig3.update_layout(
        plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
        font_color="#e8edf5", height=400, yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Call-to-action navigation ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Explore EduSentinel")
a, b, c = st.columns(3)
with a:
    st.info("**🗺️ Risk Intelligence Map**\nInteractive hotspot map across all 774 LGAs with ML-driven risk scores and driver breakdown.")
with b:
    st.warning("**⚠️ Dropout Early Warning**\nScore individual children for dropout risk with SHAP-powered explanations for field coordinators.")
with c:
    st.success("**🤖 AI Learning Assistant**\nMultilingual educational chatbot covering Nigeria's NERDC curriculum — English, Hausa, Yoruba, Igbo, Pidgin.")
