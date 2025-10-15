import os
import sys
import sqlite3
from pathlib import Path
from typing import Any, List, Tuple, TypedDict, Annotated

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# ---------- Page config ----------
st.set_page_config(page_title="Text→SQL (LangGraph)", page_icon="🎧", layout="wide")

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

# ---------- State Definition ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sql_executed: str | None
    
# ---------- Helpers ----------
def extract_sql_from_messages(messages: list) -> str | None:
    """Extract SQL from tool messages."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # Check if this was a query execution
            if "sql_db_query" in msg.name:
                # The tool input is stored in the corresponding AI message
                for m in reversed(messages):
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        for tc in m.tool_calls:
                            if 'sql_db_query' in tc.get('name', ''):
                                return tc.get('args', {}).get('query', '')
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
    engine = db._engine
    return db, engine


def create_sql_tools(db: SQLDatabase, top_k: int) -> list:
    """Create SQL tools for the agent."""
    
    def list_tables_func(_: str = "") -> str:
        """List all available tables in the database."""
        return db.get_usable_table_names()
    
    def get_schema_func(table_names: str) -> str:
        """Get the schema and sample rows for specified tables. 
        Input should be a comma-separated list of table names."""
        tables = [t.strip() for t in table_names.split(",")]
        return db.get_table_info_no_throw(tables)
    
    def query_db_func(query: str) -> str:
        """Execute a SQL query against the database and return results.
        Only SELECT queries are allowed. Returns results as a string."""
        # Safety check
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return "Error: Only SELECT queries are allowed for safety."
        
        try:
            result = db.run_no_throw(query, fetch="all")
            return str(result)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    tools = [
        Tool(
            name="sql_db_list_tables",
            func=list_tables_func,
            description="List all tables in the database. No input needed."
        ),
        Tool(
            name="sql_db_schema",
            func=get_schema_func,
            description="Get schema and sample rows for tables. Input: comma-separated table names."
        ),
        Tool(
            name="sql_db_query",
            func=query_db_func,
            description=f"Execute a SELECT query. Return at most {top_k} rows. Input: SQL query string."
        ),
    ]
    
    return tools


@st.cache_resource(show_spinner=False)
def make_graph(_db: SQLDatabase, _llm, top_k: int, prefix: str):
    """Create the LangGraph agent."""
    
    # Create tools
    tools = create_sql_tools(_db, top_k)
    
    # Bind tools to LLM
    llm_with_tools = _llm.bind_tools(tools)
    
    # Define agent node
    def agent_node(state: AgentState):
        messages = state["messages"]
        # Add system message if first turn
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            system_msg = HumanMessage(content=prefix)
            messages = [system_msg] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Define routing function
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are no tool calls, we're done
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return "end"
        return "continue"
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile
    return workflow.compile()


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
st.sidebar.write("**Framework:** LangGraph")

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
st.title("🎧 Text→SQL Agent (LangGraph)")
st.caption("Ask questions in natural language. The agent generates **SELECT** queries, executes them, and explains the result.")

llm = make_llm(model, temperature)
db, engine = make_sql_objects(db_url)
graph = make_graph(db, llm, top_k, SAFE_PREFIX_TMPL.format(top_k=top_k))

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
                    # Create initial state
                    initial_state = {
                        "messages": [HumanMessage(content=prompt)],
                        "sql_executed": None
                    }
                    
                    # Run the graph
                    result = graph.invoke(initial_state)
                    
                    # Get the final AI message
                    final_messages = result["messages"]
                    answer = ""
                    for msg in reversed(final_messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            answer = msg.content
                            break
                    
                    st.markdown(answer if answer else "_(No answer returned)_")
                    
                    # Extract and display SQL
                    sql = extract_sql_from_messages(final_messages)
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
                    
                    # Store for chat history
                    st.session_state.chat.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

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