# lesson_02g.py — Resume Parser (Real-World Application)

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

# ─── Define the structured output ───────────────────────────
class Resume(BaseModel):
    full_name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(description="Email address if present")
    years_experience: int = Field(description="Total years of experience")
    skills: List[str] = Field(description="List of technical skills")
    current_role: str = Field(description="Current or most recent job title")
    education: str = Field(description="Highest education qualification")
    summary: str = Field(description="2-line professional summary")


# ─── Step 2: Create the parser ──────────────────────────────
parser = PydanticOutputParser(pydantic_object=Resume)

# ─── Step 3: Build the prompt WITH format instructions ──────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert HR professional and resume analyzer.
     Extract structured information from resumes accurately.
     {format_instructions}"""),
    ("human", "Parse this resume:\n\n{resume_text}")
]).partial(format_instructions=parser.get_format_instructions())

# ─── Build the chain ────────────────────────────────────────
chain = prompt | llm | parser

# ─── Test with a sample resume ───────────────────────────────
sample_resume = """
Amit Sharma
amit.sharma@email.com | LinkedIn: linkedin.com/in/amit

EXPERIENCE:
Senior Software Engineer at TCS (2020-Present) — 4 years
Software Engineer at Infosys (2018-2020) — 2 years

SKILLS: Python, Django, FastAPI, PostgreSQL, Docker, AWS, REST APIs, Git

EDUCATION: B.Tech Computer Science, IIT Delhi, 2018

I specialize in building scalable backend systems and APIs.
I have 6 years of total experience in backend development.
"""

result = chain.invoke({"resume_text": sample_resume})

print("=" * 50)
print("PARSED RESUME")
print("=" * 50)
print(f"Name:         {result.full_name}")
print(f"Email:        {result.email}")
print(f"Experience:   {result.years_experience} years")
print(f"Role:         {result.current_role}")
print(f"Education:    {result.education}")
print(f"Skills:       {', '.join(result.skills)}")
print(f"Summary:      {result.summary}")
