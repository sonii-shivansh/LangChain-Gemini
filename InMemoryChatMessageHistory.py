from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)
parser = StrOutputParser()

# ─── Step 1: Build a stateless LCEL chain (as usual) ────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant named Jarvis. Be helpful and concise."),
    MessagesPlaceholder(variable_name="history"),    # ← History injected HERE
    ("human", "{input}")                            # ← Current message
])

chain = prompt | llm | parser

# ─── Step 2: Create a session store (in-memory dict) ─────────
# This stores history per session_id (one per user/conversation)
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return existing history or create a new one for this session"""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


# ─── Step 3: Wrap chain with memory ──────────────────────────
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,         # function to get/create history
    input_messages_key="input",  # which key holds the user message
    history_messages_key="history"  # which placeholder to fill
)


# ─── Step 4: Chat! Always pass session_id in config ──────────
def chat(message: str, session_id: str = "default") -> str:
    response = chain_with_memory.invoke(
        {"input": message},
        config = {"configurable": {"session_id": session_id}}
        #        ↑ THIS is how LangChain knows which history to use
    )
    return response


# ── Conversation 1 (session: rahul) ──────────────────────────
print("=== Rahul's Conversation ===")
print("Jarvis:", chat("Hi! My name is Rahul. I'm a Python developer.", "rahul_001"))
print("Jarvis:", chat("What's my name?", "rahul_001"))   # Remembers Rahul!
print("Jarvis:", chat("What's my profession?", "rahul_001"))  # Remembers Python developer!

# ── Conversation 2 (session: priya) — completely separate! ───
print("\n=== Priya's Conversation ===")
print("Jarvis:", chat("Hello! I'm Priya, I work in Data Science.", "priya_002"))
print("Jarvis:", chat("What do I do for work?", "priya_002"))  # Remembers Data Science!

# ── Cross-session isolation ───────────────────────────────────
print("\n=== Cross-Session Test ===")
print("Jarvis (Rahul's session):", chat("Who am I again?", "rahul_001"))
# Still says Rahul — sessions are isolated!

# ─── Inspect the stored history ──────────────────────────────
print("\n=== Rahul's Stored History ===")
history = session_store["rahul_001"]
for msg in history.messages:
    print(f"  [{type(msg).__name__}]: {msg.content[:60]}")