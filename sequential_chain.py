from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Build the chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an expert {domain} tutor."),
    ("human", "Explain {concept} in simple words with an analogy.")
])

parser = StrOutputParser()

# The chain — read left to right like a recipe
chain = prompt | llm | parser
#       ↑         ↑     ↑
#  formats    generates  extracts
#  messages   response   string


# ─── 4 ways to call ANY chain ───────────────────────────────

# 1. .invoke()  — one input, one output (most common)
# result = chain.invoke({
#     "domain": "Machine Learning",
#     "concept": "neural networks"
# })
# print("=== .invoke() ===")
# print(result)


# # 2. .stream() — returns tokens one by one (for real-time UI)
# print("\n=== .stream() — token by token ===")
# for token in chain.stream({"domain": "Physics", "concept": "gravity"}):
#     print(token, end="", flush=True)  # prints as it generates!
# print()  # newline at end


# 3. .batch() — multiple inputs at once (faster than loop)
print("\n=== .batch() — multiple inputs ===")
inputs = [
    {"domain": "Biology", "concept": "DNA"},
    {"domain": "Chemistry", "concept": "atoms"},
    {"domain": "History", "concept": "the Mughal Empire"},
]
results = chain.batch(inputs)
for i, res in enumerate(results):
    print(f"\nResult {i+1}: {res[:100]}...")  # first 100 chars
