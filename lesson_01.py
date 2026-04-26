from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.7
)

response = llm.invoke("What is LangChain. Explain in simple way. Why and What")

print(response.content)