# lesson_02d.py — StrOutputParser

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

parser = StrOutputParser()

# Build the chain: prompt → llm → parser
chain = prompt | llm | parser

# .invoke() now returns a plain string, not AIMessage!
result = chain.invoke({"question": "What is Python?"})

print(type(result))    # <class 'str'>  ← clean string!
print(result)          # "Python is a high-level programming language..."

# Compare — WITHOUT parser:
raw_chain = prompt | llm
raw_result = raw_chain.invoke({"question": "What is Python?"})
print(type(raw_result))  # <class 'langchain_core.messages.ai.AIMessage'>
print(raw_result.content)  # Need .content to get text