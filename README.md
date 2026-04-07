# 🤖 AI Agent with Tools - Beginner's Guide

Welcome! This is a **simple, working example** of an AI agent - think of it as a "Hello World" program for AI.

## 🎯 What is an AI Agent?

It is a program that:
- Listens to what you ask
- Decides what action to take
- Uses tools to get answers
- Explains what it did

Imagine a helpful assistant that doesn't just answer from memory - it can *search the web*, *calculate numbers*, and *check the time*.

## 🚀 Quick Start (5 minutes)

### 1. **Install Requirements**
```bash
pip install -r requirements.txt
```

### 2. **Set Up Your API Key**
Create a `.env` file in this folder:
```
GROQ_API_KEY=your_actual_key_here
```
Get a free key from: https://console.groq.com/keys

### 3. **Run It!**
```bash
streamlit run single_agent_with_tools.py
```

Open your browser to: `http://localhost:8501`

## 💬 How to Use It

Type any question in the chat box:

### Example 1: Web Search
**You ask:** "What are the latest developments in AI agents in 2026?"
```
Agent thinks: "I need current information → use web search"
↓
Searches the web
↓
Returns latest news
```

### Example 2: Do Math
**You ask:** "Calculate 25% of 2000"
```
Agent thinks: "This is a math problem → use calculator"
↓
Calculates: 2000 * 0.25 = 500
↓
Returns the answer
```

### Example 3: Check Time
**You ask:** "What's today's date?"
```
Agent thinks: "User wants date/time → use datetime tool"
↓
Gets current date
↓
Shows: Friday, April 04, 2026 at 03:45 PM
```

## 🔧 The Three Tools

### 🌍 **Web Search**
- Searches the internet in real-time
- No API key needed
- Returns relevant results
- **Try:** "Latest news about AI", "What is quantum computing?"

### 🧮 **Calculator**
- Evaluates any math expression
- Supports: `+`, `-`, `*`, `/`, `%`, `**` (power)
- Handles complex expressions with parentheses
- **Try:** "2 + 2", "100 * 3.14", "(50 + 30) / 4"

### 🕐 **Date & Time**
- Gets current date and time
- Shows day of week
- **Try:** "What time is it?", "Today's date"

# 🧠 How AI Agents Decide Which Tool to Use

An agent evaluates three things when picking a tool:

| Component | What It Is | Why It Matters |
|---|---|---|
| **Tool Name** | Identifier (e.g., `search_web`) | First signal, gives a hint about purpose |
| **Description (Docstring)** | What the tool does & when to use it | Most important, agent relies on this most heavily |
| **Parameters** | Input the tool needs (e.g., `query: str`) | Confirms if the tool fits the current context |

> 💡 **Key insight:** The description is the most critical component.
> A poorly written docstring = agent picks the wrong tool, even if the tool itself is perfect.


## 🧠 How It Works (Behind the Scenes)

```
┌─────────────────────────────────────────────┐
│  YOU ASK A QUESTION                         │
│  "Calculate 15% of 500"                     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  AI AGENT READS YOUR QUESTION               │
│  (Uses LLM)                                 │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  AGENT DECIDES: WHICH TOOL FITS?            │
│  → "This is math → use Calculator"          │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  TOOL EXECUTES                              │
│  calculator("500 * 0.15")                   │
│  → Result: "500 * 0.15 = 75"                │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  AGENT EXPLAINS ANSWER                      │
│  "15% of 500 is 75"                         │
│  (Shows: 🔧 calculator used)                │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  YOU GET ANSWER + SEE WHAT TOOL WAS USED    │
└─────────────────────────────────────────────┘
```

This pattern is called **ReAct**: Reasoning → Acting → Result

## 📚 Understanding the Code

### Where Are the Tools Defined?
**Lines 37-63** - Look for `@tool` decorator

```python
@tool
def calculator(expression: str) -> str:
    """Tool description - tells agent when to use it"""
    try:
        result = evaluate(expression)  # Safe math evaluation
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating: {e}"
```

### Where Is the Agent Created?
**Lines 70-75** - Creates the smart decision maker

```python
agent = create_react_agent(
    model=llm,           # The AI brain (Llama)
    tools=tools,         # Available tools
    prompt="..."         # Instructions for agent
)
```

### Where Does Chat History Live?
**Lines 81-87** - Remembers conversation

```python
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = [...]
```

## 🎓 Learning Exercises

Try these to understand how the agent works:

### Exercise 1: Add a New Tool
Add a greeting tool:
```python
@tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello {name}! I'm an AI assistant. How can I help?"
```

Then ask the agent: "Say hello to Alice"

### Exercise 2: Change the Prompt
Modify line 72 to make agent behave differently:
```python
prompt="You are a sarcastic AI assistant. Use tools when needed."
```

### Exercise 3: Experiment with Queries
Try these to see tool selection in action:
- ❌ "What is 2+2?" → Agent uses calculator
- ✅ "Search Google for AI news" → Agent uses web search
- ⏰ "It's currently _____?" → Agent uses datetime

## 🐛 Troubleshooting

### "GROQ_API_KEY not found"
- Make sure `.env` file exists in the same folder
- Check the API key is correct (no spaces!)
- Restart the app: `Ctrl+C` then `streamlit run ...`

### "Search failed"
- Try a simpler query
- Check your internet connection
- DuckDuckGo might be rate limiting - wait a minute

### "Error calculating: ..."
- Check your math expression syntax
- Use parentheses for complex formulas: `(10 + 5) * 2`
- Avoid unsafe characters

## 📦 What's Installed

| Package | Purpose |
|---------|---------|
| `langchain_groq` | Connect to Llama AI |
| `langchain_core` | Build tools and agents |
| `langgraph` | Agent framework (ReAct) |
| `duckduckgo_search` | Web search |
| `numexpr` | Safe math calculations |
| `streamlit` | Web interface |

## 🔐 Security Notes

⚠️ **Keep Safe:**
- Never share your `.env` file
- Don't put API keys in the code
- Don't expose this to untrusted users on a public server

## 🎬 Next Steps

After running this, you can:

1. **Add more tools** - Try adding weather, stock prices, Wikipedia lookup
2. **Improve prompts** - Make the agent more specialized
3. **Try different models** - Switch to GPT-4, Claude, etc.
4. **Build memory** - Make the agent remember across conversations
5. **Add persistence** - Save conversations to a database

## 🤔 Common Questions

**Q: Can I add more tools?**
A: Yes! Just create a function with `@tool` decorator and add it to the `tools` list.

**Q: Why Streamlit UI?**
A: It's the easiest way for beginners to see results without building a web app from scratch.

**Q: Is this production-ready?**
A: No. It's for learning. For production, add: authentication, rate limiting, logging, error tracking.

**Q: Why Llama model?**
A: It's fast, free via Groq, and good for learning. You can swap it for other models!

## 📖 Learn More

- **LangChain Docs**: https://python.langchain.com/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **ReAct Paper**: https://arxiv.org/abs/2210.03629
- **Groq Console**: https://console.groq.com/

## 💡 Tips for Success

1. **Start simple** - Ask questions that clearly need a tool
2. **Read the output** - Notice which tool was used and why
3. **Experiment** - Change prompts, add tools, break things!
4. **Read the code** - It's only ~150 lines, very readable
5. **Ask questions** - Try weird queries to see how agent reacts

## 📝 Code Structure

```
single_agent_with_tools.py
├── Imports (AI libraries)
├── Configuration (API keys, Streamlit setup)
├── Tool Definitions (3 functions with @tool)
├── Agent Creation (ReAct agent with tools)
├── Chat Interface (Streamlit UI)
└── Message Handling (conversation loop)
```

## 🎉 You Did It!

You now have a working AI agent. This is a real, functional intelligent system - the same patterns power production AI applications.

**What you learned:**
- ✅ How agents reason
- ✅ How to define tools
- ✅ How to let AI choose actions
- ✅ How to build interactive UIs

**What's next:**
- Make it cooler (more tools!)
- Make it smarter (better prompts)
- Make it useful (real-world tools)

---

**Questions? Issues?** Feel free to connect! Happy learning! 🚀

