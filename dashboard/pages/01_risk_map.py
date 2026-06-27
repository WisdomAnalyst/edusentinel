"""
Page 1 — Out-of-School Children Risk Intelligence Map
Interactive choropleth + marker cluster map for Nigeria's 774 LGAs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Map | EduSentinel", layout="wide", page_icon="🗺️")

st.title("🗺️ Out-of-School Children Risk Intelligence Map")
st.markdown(
    "AI-driven analysis of out-of-school children hotspots across Nigeria's "
    "36 states and 774 LGAs. Identifies dominant drivers and predicts next-term escalation."
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_lga():
    raw = Path(__file__).parent.parent.parent / "data" / "raw" / "nigeria_lga_education_indicators.csv"
    if raw.exists():
        return pd.read_csv(raw)
    st.error("LGA dataset not found. Run: `python -m data.synthetic.generate_nigeria_data`")
    return pd.DataFrame()

lga_df = load_lga()

if lga_df.empty:
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Map Filters")
    zones = st.multiselect(
        "Geopolitical Zone",
        lga_df["geopolitical_zone"].unique().tolist(),
        default=lga_df["geopolitical_zone"].unique().tolist(),
    )
    risk_tiers = st.multiselect(
        "Risk Tier",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"],
    )
    driver_filter = st.selectbox(
        "Filter by Dominant Driver",
        ["All"] + lga_df["dominant_driver"].unique().tolist(),
    )
    oos_min = st.slider(
        "Min OOS Rate", 0.0, 1.0, 0.0, 0.05, format="%.0%%"
    )

filtered = lga_df[
    lga_df["geopolitical_zone"].isin(zones) &
    lga_df["risk_tier"].isin(risk_tiers) &
    (lga_df["oos_rate"] >= oos_min)
]
if driver_filter != "All":
    filtered = filtered[filtered["dominant_driver"] == driver_filter]

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("LGAs shown", len(filtered))
k2.metric("OOS Children (filtered)", f"{filtered['oos_count'].sum():,.0f}")
k3.metric("Avg OOS Rate", f"{filtered['oos_rate'].mean():.1%}")
k4.metric("Critical LGAs", int((filtered["risk_tier"] == "Critical").sum()))

st.markdown("---")

# ── Map ───────────────────────────────────────────────────────────────────────
RISK_COLORS = {
    "Critical": "#ef4444",
    "High": "#f97316",
    "Medium": "#eab308",
    "Low": "#22c55e",
}

col_map, col_chart = st.columns([2, 1])

with col_map:
    st.subheader("Interactive Hotspot Map")
    m = folium.Map(
        location=[9.0, 8.0],
        zoom_start=6,
        tiles="CartoDB dark_matter",
    )

    for _, row in filtered.iterrows():
        color = RISK_COLORS.get(row["risk_tier"], "#6b7280")
        radius = max(4, min(20, row["oos_rate"] * 28))
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:200px">
          <b style="font-size:1.1em">{row['lga_name']}</b><br/>
          <b>State:</b> {row['state']}<br/>
          <b>OOS Rate:</b> {row['oos_rate']:.1%}<br/>
          <b>OOS Count:</b> {row['oos_count']:,}<br/>
          <b>Risk Tier:</b> <span style="color:{color}"><b>{row['risk_tier']}</b></span><br/>
          <b>Dominant Driver:</b> {row['dominant_driver']}<br/>
          <b>Poverty Rate:</b> {row['poverty_rate']:.1%}<br/>
          <b>Conflict Score:</b> {row['conflict_score']:.2f}<br/>
          <b>Dist to School:</b> {row['distance_to_school_km']:.1f} km
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['lga_name']} | {row['risk_tier']} | OOS: {row['oos_rate']:.1%}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;background:#0d2137;
                border:2px solid #2563eb;border-radius:8px;padding:10px;z-index:9999;
                font-family:sans-serif;color:#e8edf5;">
      <b>Risk Tier</b><br/>
      <span style="color:#ef4444">●</span> Critical (OOS ≥ 60%)<br/>
      <span style="color:#f97316">●</span> High (40–60%)<br/>
      <span style="color:#eab308">●</span> Medium (20–40%)<br/>
      <span style="color:#22c55e">●</span> Low (&lt; 20%)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, height=520, use_container_width=True)

with col_chart:
    st.subheader("Driver Breakdown")
    if "dominant_driver" in filtered.columns:
        driver_df = filtered["dominant_driver"].value_counts().reset_index()
        driver_df.columns = ["driver", "count"]
        fig = px.bar(
            driver_df,
            x="count",
            y="driver",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
            labels={"count": "# LGAs", "driver": "Driver"},
        )
        fig.update_layout(
            plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
            font_color="#e8edf5", showlegend=False,
            coloraxis_showscale=False, height=250,
            margin={"t": 10, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Tier Distribution")
    tier_df = filtered["risk_tier"].value_counts().reset_index()
    tier_df.columns = ["tier", "count"]
    tier_order = ["Critical", "High", "Medium", "Low"]
    tier_df["tier"] = pd.Categorical(tier_df["tier"], categories=tier_order, ordered=True)
    tier_df = tier_df.sort_values("tier")
    fig2 = px.bar(
        tier_df, x="tier", y="count",
        color="tier",
        color_discrete_map=RISK_COLORS,
        labels={"tier": "Risk Tier", "count": "# LGAs"},
    )
    fig2.update_layout(
        plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
        font_color="#e8edf5", showlegend=False, height=220,
        margin={"t": 10, "b": 10},
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── State-level heatmap ───────────────────────────────────────────────────────
st.subheader("State-Level OOS Rate Heatmap")
state_agg = filtered.groupby("state").agg(
    oos_rate=("oos_rate", "mean"),
    oos_count=("oos_count", "sum"),
    poverty_rate=("poverty_rate", "mean"),
    conflict_score=("conflict_score", "mean"),
).reset_index().sort_values("oos_rate", ascending=False)

fig3 = px.bar(
    state_agg,
    x="state",
    y="oos_rate",
    color="oos_rate",
    color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
    labels={"state": "State", "oos_rate": "OOS Rate"},
    hover_data={"oos_count": ":,", "poverty_rate": ":.1%", "conflict_score": ":.2f"},
    text=state_agg["oos_rate"].map(lambda x: f"{x:.0%}"),
)
fig3.update_traces(textposition="outside", textangle=-45)
fig3.update_layout(
    plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
    font_color="#e8edf5", height=400,
    coloraxis_showscale=False,
    xaxis={"tickangle": -45},
)
st.plotly_chart(fig3, use_container_width=True)

# ── Data table ────────────────────────────────────────────────────────────────
with st.expander("View Raw LGA Data Table"):
    display_cols = [
        "lga_name", "state", "geopolitical_zone", "risk_tier",
        "oos_rate", "oos_count", "dominant_driver",
        "poverty_rate", "conflict_score", "gender_gap", "distance_to_school_km",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("oos_rate", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("Download CSV", csv, "edusentinel_lga_risk.csv", "text/csv")
