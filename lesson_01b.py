# lesson_01b.py — Understanding AIMessage

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)


messages = [
    SystemMessage(content="You are a Python tutor. Always use simple examples."),
    HumanMessage(content="What is a list in Python?")
]

#  Or 
# response = llm.invoke([
#     ("system", "You are a friendly assistant named Jarvis."),
#     ("human", "What's your name and what can you do?")
# ])

response = llm.invoke(messages)
print(response.text)