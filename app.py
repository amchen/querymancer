import random

import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from querymancer.agent import ask, create_history
from querymancer.config import Config
from querymancer.models import create_llm
from querymancer.optimizations import DynamicComplexityRouter, optimize_query_execution
from querymancer.tools import with_sql_cursor

load_dotenv()

LOADING_MESSAGES = [
    "Consulting the ancient tomes of SQL wisdom...",
    "Casting query spells on your database...",
    "Summoning data from the digital realms...",
    "Deciphering your request into database runes...",
    "Brewing a potion of perfect query syntax...",
    "Channeling the power of database magic...",
    "Translating your words into the language of tables...",
    "Waving my SQL wand to fetch your results...",
    "Performing database divination...",
    "Aligning the database stars for optimal results...",
    "Consulting with the database spirits...",
    "Transforming natural language into database incantations...",
    "Peering into the crystal ball of your database...",
    "Opening a portal to your data dimension...",
    "Enchanting your request with SQL magic...",
    "Invoking the ancient art of query optimization...",
    "Reading between the tables to find your answer...",
    "Conjuring insights from your database depths...",
    "Weaving a tapestry of joins and filters...",
    "Preparing a feast of data for your consideration...",
]


def get_model(query: str = None) -> BaseChatModel:
    """Get appropriate LLM based on query complexity.

    Args:
        query: The user's natural language query (if provided)

    Returns:
        BaseChatModel: Configured language model for the query
    """
    if not query:
        # Default model if no query provided
        return create_llm(Config.MODEL)

    # Initialize the complexity router
    complexity_router = DynamicComplexityRouter()

    # Determine the appropriate model based on query complexity
    return complexity_router.get_appropriate_model(query)


def load_css(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="Querymancer",
    page_icon="🧙‍♂️",
)

# Apply CSS styling
try:
    load_css("assets/style.css")
except FileNotFoundError:
    st.write("CSS file not found, using default styling")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = create_history()

# Header and description
st.header("Querymancer")
st.subheader("Talk to your database using natural language")

# Display DB tables
with st.expander("Database Tables"):
    with with_sql_cursor() as cursor:
        # Get list of tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        st.write("### Available Tables")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            st.write(f"- {table} ({count} rows)")

# Chat interface
for message in st.session_state.messages:
    if type(message) is SystemMessage:
        continue
    is_user = type(message) is HumanMessage
    avatar = "🐧" if is_user else "🧙‍♂️"
    with st.chat_message("user" if is_user else "ai", avatar=avatar):
        st.markdown(message.content)

# Get user input
if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user", avatar="🐧"):
        st.session_state.messages.append(HumanMessage(prompt))
        st.markdown(prompt)
    with st.chat_message("ai", avatar="🧙‍♂️"):
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")

        # Optimize query execution using our pipeline
        optimized_query, model, optimized_history, model_params = optimize_query_execution(
            prompt, st.session_state.messages
        )

        # Generate response using the agent
        response = ask(optimized_query, optimized_history, model, max_iterations=10)

        # Update message placeholder with response
        message_placeholder.markdown(response)

    # Add response to chat history
    st.session_state.messages.append(AIMessage(response))
