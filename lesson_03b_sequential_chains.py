from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

parser = StrOutputParser()

# ─── REAL USE CASE: Blog Post Generator Pipeline ─────────────
# Step 1: Generate a blog title
# Step 2: Use that title to write the blog post
# Step 3: Use that post to write a Twitter/X thread


# Chain 1: Topic → Title
title_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative content writer."),
    ("human", "Generate ONE catchy blog title about: {topic}")
])

title_chain = title_prompt | llm | parser


# Chain 2: Title → Blog Post
post_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional blog writer. Write concisely."),
    ("human", "Write a 3-paragraph blog post for this title: {title}")
])

post_chain = post_prompt | llm | parser


# Chain 3: Blog Post → Twitter Thread
tweet_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a social media expert."),
    ("human", "Convert this blog post into a 3-tweet Twitter thread:\n\n{blog_post}")
])
tweet_chain = tweet_prompt | llm | parser


# ─── Option A: Manual chaining (explicit, easy to debug) ─────
topic = "LangChain for beginners"

title = title_chain.invoke({"topic": topic})
print("Generated Title:", title)

post    = post_chain.invoke({"title": title})
print("\nBlog Post:\n", post)

tweets  = tweet_chain.invoke({"blog_post": post})
print("\nTwitter Thread:\n", tweets)

