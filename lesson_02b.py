# lesson_02b.py — ChatPromptTemplate (The Industry Standard)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

# ─── METHOD 1: from_messages (most common and readable) ─────
prompt = ChatPromptTemplate.from_messages([
    # "system" role: tells AI HOW to behave — runs before every conversation
    ("system", "You are an expert {subject} teacher. Explain things simply."),
    # "human" role: the actual user question
    ("human", "Explain {topic} in simple words. Give a real-world example.")
])

# Step 1: See what the formatted messages look like
formatted_messages = prompt.format_messages(
    subject="Python programming",
    topic="decorators"
)

# print("=== Formatted Messages ===")
# for msg in formatted_messages:
#     print(f"  [{msg.type.upper()}]: {msg.content}")

# response = llm.invoke(formatted_messages)
# print(response.content)


# # Step 2: Invoke with variables
response = prompt | llm
result = response.invoke({
    "subject": "Java Backend",
    "topic": "Spring AI"
})

print("\n=== Response ===")
print(result.content)