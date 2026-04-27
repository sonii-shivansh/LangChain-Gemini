from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
parser = StrOutputParser()


# ─── WHAT IS RunnablePassthrough? ────────────────────────────
# Problem: You need to pass input to a prompt that has MULTIPLE variables,
# but one comes from the user (unchanged) and another needs processing.

passthrough = RunnablePassthrough()
print(passthrough.invoke("hello"))        # Output: "hello"   (unchanged!)
print(passthrough.invoke({"key": "val"})) # Output: {"key": "val"} (unchanged!)

# ─── REAL USE CASE: Question Answering with Context ──────────
# The prompt needs BOTH "question" (unchanged) AND "context" (processed)

# Simulated context lookup (in real RAG this would be a vector DB search)

def fetch_context(question: str) -> str:
    """Simulate fetching context documents"""
    # In real app: return vectorstore.similarity_search(question)
    return f"Context: LangChain was created by Harrison Chase in 2022. " \
           f"It supports Gemini, OpenAI, Anthropic and many other models. " \
           f"The latest version uses LCEL for building chains."


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the provided context."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# The chain structure:
#   Input "question" string
#     ├── RunnablePassthrough() → keeps question unchanged → "question" key
#     └── fetch_context lambda  → processes question     → "context" key
#         ↓
#   Both keys fed into qa_prompt → llm → parser


qa_chain = (
    RunnableParallel({
        "question": RunnablePassthrough(), # keep original question
        "context": lambda q: fetch_context(q)
    }) | qa_prompt | llm | parser
)

answer = qa_chain.invoke("Who created LangChain and when?")
print("\nQuestion: Who created LangChain and when?")
print("Answer:", answer)

answer2 = qa_chain.invoke("What models does LangChain support?")
print("\nQuestion: What models does LangChain support?")
print("Answer:", answer2)