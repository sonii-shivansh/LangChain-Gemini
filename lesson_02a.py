# lesson_02a.py — PromptTemplate Basics

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

# ─── BEFORE (bad way — hardcoded strings) ───────────────────
# response = llm.invoke("Tell me a joke about Python")
# response = llm.invoke("Tell me a joke about JavaScript")  # copy-paste!
# response = llm.invoke("Tell me a joke about SQL")         # again!

# ─── AFTER (good way — PromptTemplate) ──────────────────────


# Step 1: Define the template with {placeholders}
prompt = PromptTemplate.from_template("Tell me a {adjective} joke about {topic}")

# Step 2: Fill in variables and invoke the LLM
response = llm.invoke(prompt.format(adjective="funny", topic="Java"))
print("\nJava joke:", response.content)


response = llm.invoke(prompt.format(adjective="nerdy", topic="databases"))
print("\nDatabase joke:", response.content)

# ─── INSPECT what variables a template needs ────────────────
print("\nTemplate variables:", prompt.input_variables)
# Output: ['adjective', 'topic']
