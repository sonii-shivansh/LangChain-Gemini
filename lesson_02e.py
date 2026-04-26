# lesson_02e.py — JsonOutputParser

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

# Tell the LLM explicitly to return JSON
prompt = ChatPromptTemplate.from_messages([

    ("system", """
    You're a data extractor. Always respond ONLY with valid JSON. No markdown, no explanations.
    Just the raw JSON object.
    """),

    ("human", """
    Extract the following info from this text and return as JSON:
    - name (string)
    - age (integer)
    - skills (list of strings)

    Text: {text}
    """)

])

parser = JsonOutputParser()

chain = prompt | llm | parser

result = chain.invoke({
    "text": "My name is Priya. I am 28 years old. I know Python, Machine Learning, and SQL."
})

print(type(result))   # <class 'dict'>  ← Python dictionary!
print(result)
# Output: {'name': 'Priya', 'age': 28, 'skills': ['Python', 'Machine Learning', 'SQL']}

# Now you can work with it like a normal dict
print("\nName:", result["name"])
print("First skill:", result["skills"][0])
print("Is adult:", result["age"] >= 18)