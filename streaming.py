# lesson_03g.py — Streaming (Real-time output like ChatGPT)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)
parser = StrOutputParser()

chain = (
    ChatPromptTemplate([
        ("system", "You are a storyteller."),
        ("human", "Write a short story about {topic} in 5 sentences.")
    ])
    | llm
    | parser
)

# ─── Basic streaming ──────────────────────────────────────────
# print("Streaming story: ", end="")
# for chunk in chain.stream({"topic": "a robot learning to cook"}):
#     print(chunk, end="", flush=True)
# print("\n")


# ─── Streaming with token counting ───────────────────────────
print("Streaming with token count:")
full_response = ""
token_count = 0

for chunk in chain.stream({"topic": "a programmer who discovered magic"}):
    print(chunk, end="", flush=True)
    full_response += chunk
    token_count += 1

print(f"\n\n[Received {token_count} chunks, {len(full_response)} characters]")
