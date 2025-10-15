import os
import sys
import sqlite3
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# ---------- Page config ----------
st.set_page_config(page_title="Text→SQL (Chinook, SQLite)", page_icon="🎧", layout="wide")

# ---------- Constants ----------
CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)
DEFAULT_DB_PATH = Path("chinook.db")

SAFE_PREFIX_TMPL = """
You are a careful senior data analyst using SQLite.
Rules:
- READ-ONLY: generate only SELECT statements. Never write/alter data.
- Prefer accurate joins; inspect schema first when unsure.
- Return at most {top_k} rows unless asked for more.
- After running SQL, explain the answer clearly in plain English.
- Show the executed SQL in a ```sql block.
""".strip()

# ---------- Helpers ----------
def extract_sql(steps: List[tuple]) -> str | None:
    """Extract the last SQL the agent executed (from intermediate steps)."""
    for action, _obs in reversed(steps or []):
        try:
            tool = getattr(action, "tool", None) or action.get("tool")
            if tool and "sql_db_query" in str(tool):
                ti = getattr(action, "tool_input", None) or action.get("tool_input")
                if isinstance(ti, str):
                    return ti
        except Exception:
            pass
    return None


def create_chinook_db(db_path: Path) -> None:
    """Download the official Chinook SQL and create a local SQLite DB."""
    db_path = Path(db_path)
    if db_path.exists():
        return
    r = requests.get(CHINOOK_SQL_URL, timeout=60)
    r.raise_for_status()
    sql_text = r.text
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql_text)
        conn.commit()
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def load_environment():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("TEMPERATURE", "0"))
    top_k = int(os.getenv("TOP_K", "5"))
    # default to a path-resolved sqlite file in this folder
    default_sqlite_url = f"sqlite:///{DEFAULT_DB_PATH.resolve()}"
    db_url = os.getenv("DATABASE_URL", default_sqlite_url)
    return google_api_key, model, temperature, top_k, db_url


@st.cache_resource(show_spinner=False)
def make_llm(model: str, temperature: float):
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


@st.cache_resource(show_spinner=False)
def make_sql_objects(db_url: str):
    """Create LangChain SQLDatabase + expose its underlying SQLAlchemy engine."""
    db = SQLDatabase.from_uri(db_url, sample_rows_in_table_info=3)
    engine = db._engine  # type: ignore[attr-defined]
    return db, engine


@st.cache_resource(show_spinner=False)
def make_agent(_db: SQLDatabase, _llm, top_k: int, prefix: str):
    """Underscored params so Streamlit doesn't try to hash them."""
    return create_sql_agent(
        llm=_llm,
        db=_db,
        agent_type="tool-calling",
        top_k=top_k,
        prefix=prefix,
        verbose=True,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )


def list_tables(engine) -> list[str]:
    q = text("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]


def preview_table(engine, table: str, limit: int = 10) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(f"SELECT * FROM {table} LIMIT {limit}"), conn)


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")

google_api_key, model, temperature, top_k, db_url = load_environment()
st.sidebar.write("**Model:**", model)
st.sidebar.write("**TOP_K:**", top_k)
st.sidebar.write("**DB URL:**", db_url)

# One-click create local Chinook if using default sqlite
if db_url.startswith("sqlite:///"):
    if not DEFAULT_DB_PATH.exists():
        if st.sidebar.button("📥 Download & Create Chinook DB", type="primary"):
            with st.spinner("Downloading Chinook and creating DB..."):
                try:
                    create_chinook_db(DEFAULT_DB_PATH)
                    st.sidebar.success("chinook.db created ✓")
                except Exception as e:
                    st.sidebar.error(f"Failed to create DB: {e}")
    else:
        st.sidebar.info("chinook.db found ✓")

if not google_api_key:
    st.sidebar.error("Missing GOOGLE_API_KEY in .env — add it and rerun.")
    st.stop()

# ---------- Build components ----------
st.title("🎧 Text→SQL Agent (Chinook, SQLite)")
st.caption("Ask questions in natural language. The agent generates **SELECT** queries, executes them, and explains the result.")

llm = make_llm(model, temperature)
db, engine = make_sql_objects(db_url)
agent = make_agent(db, llm, top_k, SAFE_PREFIX_TMPL.format(top_k=top_k))

# ---------- Layout ----------
col_chat, col_browse = st.columns([2.2, 1.0])

# Chat area
with col_chat:
    st.subheader("💬 Ask your database")
    with st.expander("Try these", expanded=False):
        st.write(
            "- top 5 customers by total spend\n"
            "- revenue by country\n"
            "- tracks per genre (top 10)\n"
            "- albums per artist (top 10)\n"
            "- which city has the most invoices?\n"
        )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type your question...")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = agent.invoke({"input": prompt})
                    answer = res.get("output", "")
                    st.markdown(answer if answer else "_(No answer returned)_")

                    sql = extract_sql(res.get("intermediate_steps"))
                    if sql:
                        st.markdown("**Executed SQL**")
                        st.code(sql, language="sql")
                        # Optional: small preview of the result
                        try:
                            with engine.connect() as conn:
                                df = pd.read_sql_query(text(sql), conn)
                            if not df.empty:
                                st.markdown("**Preview (first 30 rows)**")
                                st.dataframe(df.head(30), use_container_width=True, height=300)
                        except Exception as e:
                            st.info(f"Could not preview results: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

# DB browser
with col_browse:
    st.subheader("🗂️ Browse tables")
    try:
        tables = list_tables(engine)
    except Exception as e:
        tables = []
        st.error(f"Could not list tables: {e}")

    if tables:
        t = st.selectbox("Table", tables, index=0)
        limit = st.slider("Rows to show", 5, 100, 10, step=5)
        try:
            df_prev = preview_table(engine, t, limit)
            st.dataframe(df_prev, use_container_width=True, height=340)
        except Exception as e:
            st.info(f"Could not preview table '{t}': {e}")
    else:
        if db_url.startswith("sqlite:///") and not DEFAULT_DB_PATH.exists():
            st.info("No database found yet. Use the sidebar to create Chinook.")
        else:
            st.info("No user tables detected.")
