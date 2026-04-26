# lesson_02h.py — Advanced Prompt Features
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ─── FEATURE 1: .partial() — Pre-fill variables ─────────────
# Useful when some variables are always fixed
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant for {company_name}. Language: {language}."),
    ("human", "{question}")
])

# Pre-fill company_name — now you only need to pass language + question
company_prompt = base_prompt.partial(company_name="TechCorp India")

result = company_prompt.format_messages(language="English", question="How do I reset my password?")
print("Partial prompt system:", result[0].content)
# "You are an AI assistant for TechCorp India. Language: English."

# ─── FEATURE 2: Inspecting template variables ────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "Tell me about {topic} in {language}.")
])
print("\nRequired variables:", prompt.input_variables)
# ['role', 'topic', 'language']

# ─── FEATURE 3: Multi-line system prompts (professional style) ─
professional_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert software architect with 20 years of experience.

Your rules:
1. Always explain concepts with real-world analogies
2. Provide code examples when relevant
3. Keep explanations under 200 words
4. If unsure about something, say so honestly

Format your answers clearly with sections when needed.
    """),
    ("human", "Question: {user_question}\nContext: {context}")
])

print("\nProfessional prompt variables:", professional_prompt.input_variables)
# ['user_question', 'context']