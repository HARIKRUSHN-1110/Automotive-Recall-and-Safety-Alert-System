"""
streamlit_app.py — Streamlit Web Application.

Run from project root:
    streamlit run src/app/streamlit_app.py
"""

import os
import sys
import warnings
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from sqlalchemy import create_engine, text
import traceback
# Make src/ importable from any working directory
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

# Page config — must be the FIRST Streamlit call

st.set_page_config(
    page_title = "AutoSafe — Recall Risk Checker",
    page_icon  = "🚗",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# Constants

DB_PATH    = os.path.join(ROOT, "data", "automotive_recall.db")
YEAR_MIN   = 2015
YEAR_MAX   = 2025
YEAR_DEFAULT = 2020

# DATABASE_URL — PostgreSQL in production, SQLite fallback locally
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DB_PATH}"
)


@st.cache_resource(show_spinner=False)
def get_engine():
    """
    SQLAlchemy engine — created once, reused for all DB queries.
    Works with both PostgreSQL (production) and SQLite (local fallback).
    """
    return create_engine(DATABASE_URL)


def db_query(sql: str, params: dict = None) -> list:
    """
    Run a SELECT query and return list of row dicts.
    Central helper so all DB calls go through one place.
    """
    engine = get_engine()
    with engine.connect() as conn:
        cursor = conn.execute(text(sql), params or {})
        return [dict(row) for row in cursor.mappings().fetchall()]


def db_scalar(sql: str, params: dict = None):
    """Run a query returning a single value (COUNT, MIN, MAX etc.)."""
    engine = get_engine()
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()


# Lazy-load model server
# Loaded once on first prediction, reused for every subsequent
# search. Loading takes ~200ms — we don't want it on startup.

@st.cache_resource(show_spinner=False)
def get_model_server():
    """
    Load and return the ModelServer singleton.
    @st.cache_resource means this runs ONCE per app session —
    the loaded model stays in memory across all user interactions.
    """
    from src.models.serve import ModelServer
    server = ModelServer()
    server.load_model()
    return server


@st.cache_resource(show_spinner=False)
def get_text_processor():
    """Load TextPreprocessor once and reuse."""
    from src.data_processing.text_processing import TextPreprocessor
    return TextPreprocessor(db_path=DB_PATH)

# Custom CSS
# Streamlit's default styling is functional but plain.
# A small amount of CSS makes it look significantly more
# professional without fighting the framework.

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 5px solid #e94560;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }

    /* Risk score cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .risk-low    { background: #0d2b1d; border-color: #2ecc71; }
    .risk-medium { background: #2b1f0d; border-color: #f39c12; }
    .risk-high   { background: #2b0d0d; border-color: #e74c3c; }

    /* Search form */
    .search-container {
        background: #f8f9fa;
        padding: 1.8rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }

    /* Metric badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green  { background: #d4edda; color: #155724; }
    .badge-orange { background: #fff3cd; color: #856404; }
    .badge-red    { background: #f8d7da; color: #721c24; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}

    /* Hide deploy button, github link, hosted by */
    .stDeployButton                    {display: none !important;}
    [data-testid="stToolbar"]          {display: none !important;}
    [data-testid="stDecoration"]       {display: none !important;}
    [data-testid="stStatusWidget"]     {display: none !important;}
    [data-testid="stMainMenuButton"]   {display: none !important;}

    /* Hide hosted with streamlit banner — mobile */
    .viewerBadge_container__r5tak     {display: none !important;}
    .viewerBadge_link__qRIco          {display: none !important;}
    #stDecoration                     {display: none !important;}
    iframe[title="streamlit_badge"]   {display: none !important;}
</style>
""", unsafe_allow_html=True)


# Database helpers — cached so they only hit DB once per session

@st.cache_data(show_spinner=False)
def get_manufacturers() -> list:
    """
    Load distinct manufacturer names from the database.
    Cached — only queries once per Streamlit session.
    """
    try:
        rows = db_query("SELECT DISTINCT make FROM complaints ORDER BY make")
        return [r["make"] for r in rows if r["make"]]
    except Exception as e:
        st.error(f"Database error: {e}")
        return []


@st.cache_data(show_spinner=False)
def get_models_for_make(make: str) -> list:
    """
    Load models for a specific manufacturer.
    Cached per make — switching makes re-queries once, then caches.
    """
    if not make:
        return []
    try:
        rows = db_query(
            "SELECT DISTINCT model FROM complaints WHERE make = :make ORDER BY model",
            {"make": make}
        )
        return [r["model"] for r in rows if r["model"]]
    except Exception as e:
        st.error(f"Database error: {e}")
        return []


@st.cache_data(show_spinner=False)
def get_complaint_count(make: str, model: str, year: int) -> int:
    """Return complaint count for a specific vehicle."""
    try:
        return db_scalar(
            "SELECT COUNT(*) FROM complaints WHERE make=:make AND model=:model AND model_year=:year",
            {"make": make, "model": model, "year": year}
        ) or 0
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def get_db_stats() -> dict:
    """Load summary stats for the sidebar."""
    try:
        complaints = db_scalar("SELECT COUNT(*) FROM complaints") or 0
        recalls    = db_scalar("SELECT COUNT(*) FROM recalls") or 0
        makes      = db_scalar("SELECT COUNT(DISTINCT make) FROM complaints") or 0
        return {"complaints": complaints, "recalls": recalls, "makes": makes}
    except Exception:
        return {"complaints": 0, "recalls": 0, "makes": 0}


# Sidebar

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🚗 AutoSafe")
        st.markdown("*AI-powered recall risk checker*")
        st.divider()

        # About
        with st.expander("About this tool", expanded=True):
            st.markdown("""
            AutoSafe uses machine learning to predict
            whether a vehicle is at risk of a future
            safety recall — **before** it's officially announced.

            Enter any make, model, and year to get an
            instant risk assessment powered by real
            complaint data from NHTSA.
            """)

        # How it work
        with st.expander("How it works"):
            st.markdown("""
            **1. Data collection**
            We ingest real complaint data from the
            US NHTSA database daily.

            **2. Pattern recognition**
            The LightGBM model analyses complaint
            text, components, severity flags and
            vehicle age to detect recall patterns.

            **3. Risk score**
            You get a 0–100 risk score:
            - 🟢 **0–39** Low risk
            - 🟡 **40–69** Medium risk
            - 🔴 **70–100** High risk

            **Model accuracy:** ROC-AUC 0.916
            """)

        # Dataset stats
        st.divider()
        st.markdown("### 📊 Dataset")
        stats = get_db_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Complaints", f"{stats['complaints']:,}")
            st.metric("Makes", stats['makes'])
        with col2:
            st.metric("Recalls", f"{stats['recalls']:,}")
            st.metric("Years", "2015–2025")

        # Data source
        st.divider()
        st.caption(
            "Data source: [NHTSA](https://api.nhtsa.gov/) "
            "public complaints & recalls database.\n\n"
            "⚠️ For informational purposes only. "
            "Always check official NHTSA records."
        )

# Header

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🚗 AutoSafe — Recall Risk Checker</h1>
        <p>
            AI-powered early warning system for automotive recalls.
            Search any vehicle to get an instant safety risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Search Form

def render_search_form() -> dict:
    """
    Render the search form and return search params if submitted.

    Returns:
        Dict with make, model, year if user clicked Search.
        None if form not yet submitted.
    """
    manufacturers = get_manufacturers()

    if not manufacturers:
        st.error(
            "⚠️ Could not load manufacturer list. "
            "Make sure the database exists and ingestion has been run."
        )
        return None

    st.markdown("### 🔍 Search a Vehicle")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_make = st.selectbox(
            "Manufacturer",
            options    = manufacturers,
            index      = manufacturers.index("BMW") if "BMW" in manufacturers else 0,
            help       = "Select the vehicle manufacturer",
            key        = "make_select",
        )

    with col2:
        models = get_models_for_make(selected_make)
        selected_model = st.selectbox(
            "Model",
            options = models if models else ["— no models found —"],
            help    = "Models are filtered by the selected manufacturer",
            key     = "model_select",
        )

    with col3:
        selected_year = st.number_input(
            "Year",
            min_value = YEAR_MIN,
            max_value = YEAR_MAX,
            value     = YEAR_DEFAULT,
            step      = 1,
            help      = f"Model year ({YEAR_MIN}–{YEAR_MAX})",
            key       = "year_input",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Year slider as a visual alternative
    selected_year_slider = st.slider(
        "Or drag to select year",
        min_value = YEAR_MIN,
        max_value = YEAR_MAX,
        value     = int(selected_year),
        key       = "year_slider",
    )

    # Sync slider and number input — slider takes priority if moved
    final_year = selected_year_slider

    # Quick stats
    if selected_make and selected_model and "no models" not in selected_model:
        complaint_count = get_complaint_count(selected_make, selected_model, final_year)
        if complaint_count > 0:
            st.caption(
                f"📋 Found **{complaint_count:,}** complaints for "
                f"{selected_make} {selected_model} {final_year} in our database."
            )
        else:
            st.caption(
                f"⚠️ No complaints found for "
                f"{selected_make} {selected_model} {final_year}. "
                f"The risk score will be based on limited data."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Search button
    col_left, col_btn, col_right = st.columns([2, 1, 2])
    with col_btn:
        search_clicked = st.button(
            "Check Risk",
            type      = "primary",
            use_container_width = True,
            key       = "search_btn",
        )

    if search_clicked and "no models" not in selected_model:
        # Store in session state so results last across reruns
        st.session_state["last_search"] = {
            "make":  selected_make,
            "model": selected_model,
            "year":  final_year,
        }
        return st.session_state["last_search"]

    # If user hasn't searched yet but has a previous search, restore it
    if "last_search" in st.session_state and not search_clicked:
        return st.session_state["last_search"]

    return None

# fetch complaints & run prediction

def get_risk_assessment(make: str, model: str, year: int) -> dict:
    """
    Full pipeline for one vehicle search:
      1. Fetch top complaints from database
      2. Concatenate their summaries into one text block
      3. Clean the text via TextPreprocessor
      4. Call ModelServer.predict_proba()
      5. Return result + raw complaints for display

    Why concatenate summaries?
    The model was trained on individual complaint summaries.
    At inference time we don't have one complaint — we have many.
    Concatenating the most recent complaints gives the model a
    rich signal that reflects the vehicle's current complaint pattern.

    Returns:
        Dict with keys: result, complaints, recalls, error
    """
    try:
        complaints = db_query(
            """
            SELECT odi_number, component, summary, crash, fire,
                   injuries, deaths, date_complained
            FROM complaints
            WHERE make = :make AND model = :model AND model_year = :year
            ORDER BY date_complained DESC
            LIMIT 20
            """,
            {"make": make, "model": model, "year": year}
        )

        recalls = db_query(
            """
            SELECT campaign_number, component, summary,
                   consequence, remedy, recall_date, park_it
            FROM recalls
            WHERE make = :make AND model = :model AND model_year = :year
            ORDER BY recall_date DESC
            """,
            {"make": make, "model": model, "year": year}
        )

        if not complaints:
            return {
                "result":     None,
                "complaints": [],
                "recalls":    recalls,
                "error":      None,
                "no_data":    True,
            }
        else:
            # Combine all summaries for richer NLP signal
            summary_text = " ".join(
                c["summary"] for c in complaints if c["summary"]
            )
            # Use the most common component
            component = complaints[0]["component"] or "UNKNOWN"
            crash     = any(c["crash"] for c in complaints)
            fire      = any(c["fire"]  for c in complaints)

        # Clean the text through the same pipeline used in training
        processor    = get_text_processor()
        clean_summary = processor.process(summary_text)

        if not clean_summary:
            clean_summary = f"{make} {model} vehicle issue"

        # Run model prediction
        server = get_model_server()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = server.predict_proba(
                make      = make,
                model     = model,
                year      = year,
                summary   = clean_summary,
                component = component,
                crash     = crash,
                fire      = fire,
            )

        return {
            "result":     result,
            "complaints": complaints,
            "recalls":    recalls,
            "error":      None,
        }

    except Exception as e:
        return {
            "result":     None,
            "complaints": [],
            "recalls":    [],
            "error":      traceback.format_exc(),
        }

# Risk gauge plotly

def render_risk_gauge(score: int, label: str):

    if score < 40:
        color = "#2ecc71"
    elif score < 70:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": f"{label} Risk", "font": {"size": 16}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  40], "color": "rgba(46,204,113,0.15)"},
                {"range": [40, 70], "color": "rgba(243,156,18,0.15)"},
                {"range": [70,100], "color": "rgba(231,76,60,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        number = {"font": {"size": 48, "color": color}},
    ))

    fig.update_layout(
        height  = 220,
        margin  = dict(t=40, b=10, l=20, r=20),
        paper_bgcolor = "rgba(0,0,0,0)",
        font_color    = "#ffffff",
    )

    st.plotly_chart(fig, use_container_width=True)

# complaints table for proof
def render_complaints_table(complaints: list):
    """Show top 10 complaints in a sortable, expandable table."""
    st.markdown("### Most Recent Complaints")

    if not complaints:
        st.info("No complaints found for this vehicle/year.")
        return

    # Build display dataframe
    rows = []
    for c in complaints[:10]:
        rows.append({
            "Date":      c.get("date_complained", "N/A") or "N/A",
            "Component": (c.get("component") or "UNKNOWN")[:40],
            "Summary":   (c.get("summary") or "")[:100] + "...",
            "Crash":     "🚨 Yes" if c.get("crash") else "No",
            "Injuries":  int(c.get("injuries") or 0),
        })
    df = pd.DataFrame(rows)

    # Sort controls
    sort_col = st.selectbox(
        "Sort by",
        ["Date", "Component", "Injuries"],
        key="complaint_sort",
    )
    df = df.sort_values(sort_col, ascending=(sort_col != "Injuries"))

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Expandable full text for each complaint
    st.markdown("#### 🔍 Full Complaint Text")
    for i, c in enumerate(complaints[:10]):
        summary = c.get("summary") or "No summary available"
        date    = c.get("date_complained") or "Unknown date"
        comp    = c.get("component") or "UNKNOWN"
        label   = f"{date} — {comp[:40]}"
        with st.expander(label):
            st.write(summary)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**Crash:** {'Yes 🚨' if c.get('crash') else 'No'}")
            with col2:
                st.caption(f"**Fire:** {'Yes 🔥' if c.get('fire') else 'No'}")
            with col3:
                st.caption(f"**Injuries:** {c.get('injuries') or 0}")

# recalls table for proof and understanding
def render_recalls_section(recalls: list, make: str, model: str, year: int):
    """Show all historical recalls for this vehicle."""
    st.markdown("### Recall History")

    if not recalls:
        st.success(f"**No recalls on record for {make} {model} {year}.**")
        return

    st.error(f"⚠️ **{len(recalls)} recall(s)** found for {make} {model} {year}")

    for rec in recalls:
        campaign = rec.get("campaign_number", "N/A")
        date     = rec.get("recall_date", "N/A")
        comp     = rec.get("component", "UNKNOWN")
        park_it  = rec.get("park_it", 0)

        # Label shows campaign # and date — enough to identify at a glance
        label = f"📌 {campaign}  |  {date}  |  {comp[:50]}"

        with st.expander(label, expanded=(len(recalls) == 1)):
            if park_it:
                st.error("🚫 NHTSA advises: **Do not drive this vehicle**")

            st.markdown(f"**Summary:** {rec.get('summary') or 'N/A'}")
            st.markdown(f"**Consequence:** {rec.get('consequence') or 'N/A'}")
            st.markdown(f"**Remedy:** {rec.get('remedy') or 'N/A'}")

            # Official NHTSA link
            nhtsa_link = f"https://www.nhtsa.gov/recalls?nhtsaId={campaign}"
            st.markdown(f"🔗 [View official NHTSA recall page]({nhtsa_link})")


@st.cache_data(show_spinner=False)
def get_complaints_for_charts(make: str, model: str, year: int) -> dict:
    """
    Fetch all complaints for this vehicle for charting.
    Separate from the 20-complaint limit used for prediction —
    charts need the full history for accurate trend lines.
    """
    try:
        rows = db_query(
            """
            SELECT date_complained, component, injuries
            FROM complaints
            WHERE make = :make AND model = :model AND model_year = :year
            ORDER BY date_complained ASC
            """,
            {"make": make, "model": model, "year": year}
        )

        siblings = db_query(
            """
            SELECT model, COUNT(*) as cnt
            FROM complaints
            WHERE make = :make AND model_year = :year
            GROUP BY model
            ORDER BY cnt DESC
            LIMIT 8
            """,
            {"make": make, "year": year}
        )

        return {
            "complaints": [{"date": r["date_complained"], "component": r["component"], "injuries": r["injuries"]} for r in rows],
            "siblings":   [{"model": r["model"], "count": r["cnt"]} for r in siblings],
        }
    except Exception:
        return {"complaints": [], "siblings": []}

def render_visualizations(make: str, model: str, year: int):
    """Three Plotly charts: trend, component breakdown, model comparison."""

    st.markdown("### Visual Analysis")

    data      = get_complaints_for_charts(make, model, year)
    complaints = data["complaints"]
    siblings   = data["siblings"]

    if not complaints:
        st.info("Not enough complaint data to generate charts for this vehicle.")
        return

    df = pd.DataFrame(complaints)

    # Chart 1: Complaint trend by month
    st.markdown("#### Complaint Volume Over Time")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_valid   = df.dropna(subset=["date"])

    if len(df_valid) >= 3:
        df_monthly = (
            df_valid.set_index("date")
            .resample("ME")
            .size()
            .reset_index(name="complaints")
        )
        fig_trend = px.line(
            df_monthly,
            x     = "date",
            y     = "complaints",
            title = f"{make} {model} {year} — Monthly Complaints",
            labels = {"date": "Month", "complaints": "Complaints Filed"},
        )
        fig_trend.update_traces(
            line_color = "#e94560",
            line_width = 2,
            mode       = "lines+markers",
        )
        fig_trend.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font_color    = "#ffffff",
            height        = 300,
            margin        = dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption(
            "A sudden spike in complaints over a short period "
            "is a strong early warning signal for an upcoming recall."
        )
    else:
        st.info("Not enough dated complaints to show a trend line.")

    # Charts 2 & 3 side by side
    col_pie, col_bar = st.columns(2)

    # Chart 2: Component breakdown pie
    with col_pie:
        st.markdown("#### Component Breakdown")
        components = [
            (c.get("component") or "UNKNOWN").split(",")[0].strip()
            for c in complaints
        ]
        comp_counts = Counter(components).most_common(8)

        if comp_counts:
            labels = [c[0] for c in comp_counts]
            values = [c[1] for c in comp_counts]
            fig_pie = go.Figure(go.Pie(
                labels    = labels,
                values    = values,
                hole      = 0.4,
                textinfo  = "percent+label",
                hoverinfo = "label+value+percent",
            ))
            fig_pie.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                font_color    = "#ffffff",
                height        = 480,
                margin        = dict(t=20, b=80, l=20, r=20),
                showlegend    = True,
                legend        = dict(
                    orientation = "h",
                    yanchor     = "bottom",
                    y           = -0.50,
                    xanchor     = "center",
                    x           = 0.5,
                    font        = dict(size=12),
                ),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption(
                "Components like AIR BAGS and FUEL SYSTEM "
                "have historically higher recall rates than others."
            )

    #Chart 3: Comparison to similar models
    with col_bar:
        st.markdown("#### vs Similar Models")
        if siblings:
            sib_df = pd.DataFrame(siblings)
            # Highlight the searched model
            sib_df["color"] = sib_df["model"].apply(
                lambda m: "#e94560" if m == model else "#4a90d9"
            )
            fig_bar = go.Figure(go.Bar(
                x           = sib_df["count"],
                y           = sib_df["model"],
                orientation = "h",
                marker_color = sib_df["color"],
                hovertemplate = "%{y}: %{x} complaints<extra></extra>",
            ))
            fig_bar.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                font_color    = "#ffffff",
                height        = 320,
                margin        = dict(t=20, b=20, l=20, r=20),
                xaxis_title   = "Total Complaints",
                yaxis_title   = "",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(
                f"Red bar = {model} (searched vehicle). "
                "Blue bars = other models from same make and year."
            )

# Risk Assessment Display

def render_risk_assessment(search: dict):
    """
    Full risk assessment section shown after a search.
    Calls the backend, renders the gauge, and shows key metrics.
    """
    make  = search["make"]
    model = search["model"]
    year  = search["year"]

    st.divider()
    st.markdown(f"### 📊 Risk Assessment — {make} {model} {year}")

    # Run prediction
    with st.spinner("Analysing complaints and predicting recall risk..."):
        data = get_risk_assessment(make, model, year)
    if data.get("no_data"):
        st.warning(
            f"⚠️ No complaints found for {make} {model} {year} in our database. "
            f"We cannot generate a meaningful risk score without complaint data. "
            f"This vehicle may not have been included in our ingestion, or may have "
            f"genuinely very few complaints filed with NHTSA."
        )
        if data["recalls"]:
            st.error(f"⚠️ However, {len(data['recalls'])} recall(s) are on record.")
        return

    if data["error"]:
        st.error(f"⚠️ Prediction error: {data['error']}")
        return

    result     = data["result"]
    complaints = data["complaints"]
    recalls    = data["recalls"]

    #Three-column layout
    col_gauge, col_metrics, col_context = st.columns([1.2, 1, 1])

    #Left: Gauge
    with col_gauge:
        render_risk_gauge(result.risk_score, result.risk_label)

    #Middle: Key metrics
    with col_metrics:
        st.markdown("#### Key Metrics")

        # Risk probability bar
        st.markdown("**Recall probability**")
        st.progress(result.probability)
        st.caption(f"{result.probability*100:.1f}% chance of recall")

        st.markdown("---")

        # Complaint count
        complaint_count = len(complaints)
        st.metric("Complaints analysed", complaint_count)

        # Severity flags
        crash_count = sum(1 for c in complaints if c.get("crash"))
        fire_count  = sum(1 for c in complaints if c.get("fire"))
        injury_count = sum(c.get("injuries", 0) or 0 for c in complaints)

        if crash_count or fire_count or injury_count:
            st.markdown("**Severity signals detected**")
            if crash_count:
                st.error(f"🚨 {crash_count} crash report(s)")
            if fire_count:
                st.error(f"🔥 {fire_count} fire report(s)")
            if injury_count:
                st.warning(f"🏥 {injury_count} injur(ies) reported")

    # Right: Context
    with col_context:
        st.markdown("#### Context")

        # Existing recalls
        if recalls:
            st.error(f"⚠️ **{len(recalls)} active recall(s) on record**")
            for rec in recalls[:2]:
                with st.expander(f"Recall: {rec['campaign_number']}"):
                    st.write(f"**Component:** {rec['component']}")
                    st.write(f"**Date:** {rec['recall_date']}")
                    st.write(rec['summary'][:300] + "..." if rec['summary'] and len(rec['summary']) > 300 else rec.get('summary',''))
                    if rec.get("park_it"):
                        st.error("🚫 NHTSA advises: Do not drive this vehicle")
        else:
            st.success("No active recalls on record for this vehicle")

        st.markdown("---")

        # Model confidence note
        st.markdown("**About this score**")
        st.caption(
            f"Score based on {complaint_count} complaint(s). "
            f"Model threshold: {result.threshold:.3f}. "
            f"Inference time: {result.elapsed_ms:.0f}ms."
        )

        if complaint_count == 0:
            st.warning(
                "⚠️ No complaints found for this vehicle/year. "
                "Score is based on minimal data — interpret with caution."
            )
        elif complaint_count < 5:
            st.warning("⚠️ Limited data — score may be less reliable.")

    #Risk interpretation
    st.markdown("<br>", unsafe_allow_html=True)

    if result.risk_label == "High":
        st.error(
            "🔴 **High Risk** — This vehicle's complaint pattern resembles "
            "vehicles that were subsequently recalled. We recommend checking "
            "the official NHTSA database and contacting your dealer."
        )
    elif result.risk_label == "Medium":
        st.warning(
            "🟡 **Medium Risk** — Some elevated complaint signals detected. "
            "Monitor this vehicle for new complaints and stay alert for "
            "official recall notices."
        )
    else:
        st.success(
            "**Low Risk :** Complaint pattern does not strongly resemble "
            "recalled vehicles. Continue to check NHTSA periodically."
        )

    #NHTSA link
    nhtsa_url = (
        f"https://www.nhtsa.gov/vehicle/{make}/{model}/{year}/4DR"
    )
    st.markdown(
        f"🔗 [Check official NHTSA records for {make} {model} {year}]({nhtsa_url})",
        unsafe_allow_html=False,
    )

    render_complaints_table(complaints)
    st.divider()
    render_recalls_section(recalls, make, model, year)
    st.divider()
    render_visualizations(make, model, year)

# Main

def main():
    render_sidebar()
    render_header()
    render_search_form_result = render_search_form()

    if render_search_form_result:
        render_risk_assessment(render_search_form_result)
    else:
        # Landing state — no search yet
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #888;">
            <div style="font-size: 4rem;">🚗</div>
            <h3 style="color: #444;">Select a vehicle above to check its recall risk</h3>
            <p>Search any make, model, and year from our database of 179,000+ complaints</p>
        </div>
        """, unsafe_allow_html=True)

        # Show example vehicles as inspiration
        st.markdown("#### 💡 Try these examples:")
        examples = [
            ("BMW",     "X5",      2019),
            ("TOYOTA",  "Camry",   2020),
            ("FORD",    "F-150",   2018),
            ("TESLA",   "Model 3", 2021),
        ]

        cols = st.columns(len(examples))
        for col, (make, model, year) in zip(cols, examples):
            with col:
                if st.button(
                    f"{make}\n{model} {year}",
                    key             = f"example_{make}_{model}_{year}",
                    use_container_width = True,
                ):
                    st.session_state["last_search"] = {
                        "make":  make,
                        "model": model,
                        "year":  year,
                    }
                    st.rerun()

if __name__ == "__main__":
    main()