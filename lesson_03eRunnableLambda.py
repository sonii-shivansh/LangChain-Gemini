# lesson_03e.py — RunnableLambda

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
parser = StrOutputParser()

# ─── Basic RunnableLambda examples ───────────────────────────
# Wrap any Python function as a chain step

# clean_text = RunnableLambda(lambda x: x.strip().lower())
# count_words = RunnableLambda(lambda x: len(x.split()))
# add_prefix   = RunnableLambda(lambda x: f"User said: {x}")

# print(clean_text.invoke("  Hello World  "))  # "hello world"
# print(count_words.invoke("one two three"))   # 3
# print(add_prefix.invoke("I love Python"))    # "User said: I love Python"


# ------------------------------------------------------------------------------------------------------------
# ─── REAL USE CASE: Pre-process + Post-process in pipeline ───

def preprocess_input(text: str) -> dict:
    """Clean and structure user input"""
    return {
        "cleaned_text": text.strip(),
        "word_count": len(text.split()),
        "has_question": "?" in text
    }

def format_output(response: str) -> str:
    """Post-process LLM output"""
    lines = response.strip().split("\n")
    formatted = "\n".join(f"  → {line}" for line in lines if line.strip())
    return f"ANSWER:\n{formatted}"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    ("human", "{cleaned_text}")
])

full_chain = (
    RunnableLambda(preprocess_input)   # Step 1: pre-process input
    | prompt                            # Step 2: format into messages
    | llm                               # Step 3: call Gemini
    | parser                            # Step 4: extract string
    | RunnableLambda(format_output)    # Step 5: post-process output
)


result = full_chain.invoke("What are the 3 main benefits of using Python?")
print(result)