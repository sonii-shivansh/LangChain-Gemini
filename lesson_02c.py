from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5
)

# ─── 1. SystemMessage ────────────────────────────────────────
# Sets the AI's "personality" and "rules" for the ENTIRE conversation
# Think of it as the employee handbook given to a new hire

system = SystemMessage(content="""
You are Jarvis, a sarcastic-but-helpful AI assistant.
You always give correct answers but with a hint of sarcasm.
You end every response with a relevant emoji.
""")

# ─── 2. HumanMessage ─────────────────────────────────────────
# This is what the user sends each time
human = HumanMessage(content="What is 2 + 2?")

response = llm.invoke([system, human])
print("With SystemMessage persona:")
print(response.content)

# ─── 3. AIMessage (for conversation history) ─────────────────
# When you want to include PAST conversation, use AIMessage
# This is how you simulate multi-turn conversation

conversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="My name is Rahul."),
    AIMessage(content="Nice to meet you, Rahul! How can I help you today?"),
    HumanMessage(content="What's my name?")   # AI should remember "Rahul"
]

response2 = llm.invoke(conversation)
print("\nWith conversation history (AIMessage):")
print(response2.content)
# Should say: "Your name is Rahul!"
