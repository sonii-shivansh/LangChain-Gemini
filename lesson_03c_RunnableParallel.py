from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)
parser = StrOutputParser()

# ─── REAL USE CASE: Analyze an article from 3 angles at once ─

summary_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a summarizer."),
        ("human", "Summarize this in 2 sentences:\n{article}")
    ]) | llm | parser
)

sentiment_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment analyst."),
        ("human", "Is this text positive, negative or neutral? Explain briefly:\n{article}")
    ]) | llm | parser
)

keywords_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a keyword extractor."),
        ("human", "List 5 key topics from this text as a comma-separated list:\n{article}")
    ]) | llm | parser
)

# ─── Run all 3 IN PARALLEL (3x faster than one by one!) ─────
parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "keywords":  keywords_chain
})


article = """
LangChain has revolutionized how developers build AI applications.
Released in 2022, it quickly became the most popular framework for
working with Large Language Models. Its modular design allows
developers to mix and match components like prompts, memory, and
agents. The community has grown to hundreds of thousands of users
and the framework powers production systems at major companies.
"""

# This makes 3 API calls simultaneously — not one after another!
result = parallel_chain.invoke({"article": article})

print("=== PARALLEL RESULTS ===")
print("Summary:\n", result["summary"])
print("\nSentiment:\n", result["sentiment"])
print("\nKeywords:\n", result["keywords"])

# result is a plain Python dict — use it freely
print("\n--- Accessing results ---")
print("Type:", type(result))        # <class 'dict'>
print("Keys:", list(result.keys())) # ['summary', 'sentiment', 'keywords']