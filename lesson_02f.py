# lesson_02f.py — PydanticOutputParser (Professional Grade)


from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

# ─── Step 1: Define the data structure you WANT ─────────────
class MovieReview(BaseModel):
    title: str = Field(description="Name of the movie")
    rating: float = Field(description="Rating out of 10")
    genre: List[str] = Field(description="List of genres")
    summary: str = Field(description="One sentence summary of the plot")
    recommended: bool = Field(description="Whether you recommend the movie")


# ─── Step 2: Create the parser ──────────────────────────────
parser = PydanticOutputParser(pydantic_object=MovieReview)

# This is KEY: the parser generates instructions for the LLM
format_instructions = parser.get_format_instructions()
print("=== Format Instructions (sent to LLM) ===")
print(format_instructions[:500])  # Shows schema instructions

# ─── Step 3: Build the prompt WITH format instructions ──────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic. {format_instructions}"),
    ("human", "Write a review for the movie: {movie_name}")
]).partial(format_instructions=format_instructions)
# .partial() pre-fills a variable so you don't need to pass it every time

# ─── Step 4: Build and invoke the chain ─────────────────────
chain = prompt | llm | parser

result = chain.invoke({"movie_name": "Inception"})

# result is now a MovieReview OBJECT — not a string or dict!
print(type(result))           # <class '__main__.MovieReview'>
print("\nTitle:", result.title)
print("Rating:", result.rating)
print("Genres:", result.genre)
print("Summary:", result.summary)
print("Recommended:", result.recommended)

# Works like a real Python object
if result.rating > 8.0:
    print(f"\n{result.title} is a must-watch!")