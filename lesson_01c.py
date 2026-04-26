# lesson_01c.py — Exploring LLM Parameters

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


# TEMPERATURE — Controls creativity/randomness
# 0.0 = very focused, deterministic (good for facts, code)
# 0.7 = balanced (good for general use)
# 1.0 = very creative (good for stories, brainstorming)

# Low temperature example — same answer every time
llm_precise = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0
)

# High temperature example — different answer each time
llm_creative = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0 # Gemini 3.0+ defaults to 1.0
)

question = "Give me one word that describes the sky."

print("Precise (temp=0):", llm_precise.invoke(question).content)
print("Creative (temp=1):", llm_creative.invoke(question).content)