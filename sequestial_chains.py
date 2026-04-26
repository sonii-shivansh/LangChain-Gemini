from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

explain_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} simply"
)

example_prompt = ChatPromptTemplate.from_template(
    "Give 3 examples of: {text}"
)

chain = (
    explain_prompt
    | llm
    | StrOutputParser()
    | example_prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "REST API"})
print(result)