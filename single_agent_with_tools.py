import os
import datetime
import streamlit as st
from dotenv import load_dotenv
 
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from numexpr import evaluate

# LOAD .env FILE
# ──────────────────────────────────────────────
load_dotenv()
 
# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.header("🤖 AI Agent with Tools")
st.write("An agent that can search the web, do math, and more")

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    streaming=True,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# -------------- Define custom tools with @tool decorator ---------------

# Tool 1: Web Search (uses DuckDuckGo, no API key needed)
_search = DuckDuckGoSearchRun()
@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this for any questions about recent events, news, or facts you don't know."""
    try:
        result = _search.run(query)
        return result[:2000]  # trim long results to avoid token limits
    except Exception as e:
        return f"Search failed: {e}. Please try a different query."
 
# Tool 2: Calculator - For ANY math calculation
@tool
def calculator(expression: str) -> str:
    """ALWAYS use this tool for ANY math calculation, no matter how simple.
    Example inputs: '2 + 2', '100 * 0.15', '(50 + 30) / 4', '1+1-2+3%4'
    """
    try:
        result = evaluate(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating: {e}"

# Tool 3: Get current date/time
@tool
def get_current_datetime() -> str:
    """Get the current date and time. Use this when user asks about today's date or current time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")
 
# Collect all tools in a list
tools = [web_search, calculator, get_current_datetime]

# Show available tools in sidebar
st.sidebar.header("Available Tools")
for t in tools:
    st.sidebar.write(f"🔧 **{t.name}**")

# CREATE THE AGENT
# ──────────────────────────────────────────────
# create_react_agent = LangGraph's built-in agent
# It uses the ReAct pattern: Reason → Act → Observe → Repeat
# The LLM decides WHICH tool to use based on the question
 
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful AI assistant. Use your tools when needed. Always be concise.",
)

# CHAT HISTORY
# ──────────────────────────────────────────────
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = [
        {"role": "assistant", "content": "Hi! I can search the web, do math, and tell you the date. What do you need?"}
    ]
 
for msg in st.session_state.agent_messages:
    st.chat_message(msg["role"]).write(msg["content"])
 
# HANDLE USER INPUT
# ──────────────────────────────────────────────
if user_query := st.chat_input("Ask me anything..."):
 
    # 1. Show user message
    st.session_state.agent_messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # 2. Build message history for the agent
    langchain_messages = []
    for msg in st.session_state.agent_messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    # 3. Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"messages": langchain_messages})
                # Show tools used in the UI
                tools_used = []
                for msg in result["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tools_used.append(f"🔧 **{tool_call['name']}** → `{tool_call['args']}`")
                if tools_used:
                    st.info("**Tools used:**\n" + "\n".join(tools_used))
                else:
                    st.info("💬 No tools used - answered from model knowledge")
                final_message = result["messages"][-1].content
            except Exception as e:
                final_message = f"Sorry, something went wrong. Please try rephrasing your question. (Error: {str(e)[:100]})"

            st.write(final_message)

    # 4. Save response
    st.session_state.agent_messages.append({"role": "assistant", "content": final_message})
