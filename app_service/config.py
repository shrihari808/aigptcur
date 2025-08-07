import chromadb
from chromadb.config import Settings,DEFAULT_DATABASE,DEFAULT_TENANT
import os
from langchain_openai import ChatOpenAI
import chromadb.utils.embedding_functions as embedding_functions
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import tiktoken
from pathlib import Path
import numpy as np

load_dotenv(override=True)

#!----THIS IS FOR MARKETDATALM SERVICE----!
import os
from dotenv import load_dotenv
from pathlib import Path
import tiktoken

# --- CORRECTED: Load environment variables from the root .env file ---
# This code constructs the correct path to your .env file, which should be
# in the root 'aigptcur' directory, one level above this 'app_service' directory.
try:
    env_path = Path(__file__).resolve().parent.parent / '.env'
    print(f"INFO:     Attempting to load environment variables from: {env_path}")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("INFO:     .env file found and loaded.")
    else:
        print("ERROR:    .env file NOT FOUND at the expected path. Please ensure it exists.")
except Exception as e:
    print(f"ERROR:    Could not load .env file: {e}")


# --- Configuration variables from your original project ---
# (Add any other original config variables you had here)


# --- NEW: Configuration variables merged from MarketDataLM ---

# API Keys
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Configuration
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "market-data-index")

# --- ADDED: Diagnostic prints to check if variables were loaded ---
print("\n--- DIAGNOSTIC: Checking loaded environment variables ---")
print(f"PINECONE_API_KEY loaded: {'Yes' if PINECONE_API_KEY else 'NO - THIS IS THE PROBLEM'}")
print(f"PINECONE_ENVIRONMENT loaded: {'Yes' if PINECONE_ENVIRONMENT else 'NO - THIS IS THE PROBLEM'}")
print(f"PINECONE_INDEX_NAME loaded: {PINECONE_INDEX_NAME}") # This has a default, so it should appear
print(f"BRAVE_API_KEY loaded: {'Yes' if BRAVE_API_KEY else 'NO - THIS IS THE PROBLEM'}")
print("-------------------------------------------------------\n")


# OpenAI Model Definitions
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Tokenizer for 'text-embedding-ada-002'
encoding = tiktoken.get_encoding("cl100k_base")

# Brave Search Parameters
MAX_SCRAPED_SOURCES = 30
MAX_PAGES = 1
MAX_ITEMS_PER_DOMAIN = 1

# Content Processing Limits
MAX_WEBPAGE_CONTENT_TOKENS = 1000
MAX_EMBEDDING_TOKENS = 8000
MAX_RERANKED_CONTEXT_ITEMS = 10

# Pinecone Indexing Wait Constants
PINECONE_MAX_WAIT_TIME = 30
PINECONE_CHECK_INTERVAL = 1

# Context Sufficiency Assessment
CONTEXT_SUFFICIENCY_THRESHOLD = 0.3
MIN_CONTEXT_LENGTH = 200  # Minimum characters in context
MIN_RELEVANT_DOCS = 3     # Minimum number of relevant documents

# Re-ranking Weight Constants
W_RELEVANCE = float(os.getenv("W_RELEVANCE", 0.5450))
W_SENTIMENT = float(os.getenv("W_SENTIMENT", 0.1248))
W_TIME_DECAY = float(os.getenv("W_TIME_DECAY", 0.2814))
W_IMPACT = float(os.getenv("W_IMPACT", 0.0488))

# Source Credibility Weights
SOURCE_CREDIBILITY_WEIGHTS = {
    "moneycontrol.com": 0.9,
    "economictimes.indiatimes.com": 0.9,
    "business-standard.com": 0.85,
    "livemint.com": 0.85,
    "cnbctv18.com": 0.8,
    "screener.in": 0.95,
    "trendlyne.com": 0.9,
    "bloomberg.com": 0.95,
    "reuters.com": 0.95,
    "financialexpress.com": 0.85,
    "thehindubusinessline.com": 0.8,
    "ndtv.com": 0.75,
    "zeebiz.com": 0.7,
    "businesstoday.in": 0.8,
    "default": 0.5
}

# Impact Keywords for Scoring
IMPACT_KEYWORDS = [
    "price change", "rating downgrade", "layoffs", "policy changes",
    "acquisition", "merger", "earnings surprise", "bankruptcy",
    "restructuring", "dividend cut", "share buyback", "new product launch",
    "regulatory approval", "legal dispute", "fraud", "scandal",
    "inflation", "recession", "interest rate", "gdp growth", "unemployment",
    "high", "spike", "surge", "plunge", "soar", "crash", "rally",
    "breakout", "resistance", "support", "bullish", "bearish"
]

#CHROMA_SERVER
chroma_username=os.getenv("CHROMA_USERNAME")
chroma_password=os.getenv("CHROMA_PASSWORD")
chroma_host=os.getenv("CHROMA_HOST")

chroma_server_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=9001,
    # ... other settings
)


client=chroma_server_client
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs= Chroma(
    client=client,
    collection_name="brave_scraped",
    embedding_function=embeddings,)

vs_promoter= Chroma(
    client=client,
    collection_name="promoters_202409",  #promoters,promoters_202409
    embedding_function=embeddings,)



openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )


default_ef = embedding_functions.DefaultEmbeddingFunction()



GPT3_4k = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
GPT3_16k = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
GPT4 = ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09")
GPT4o =ChatOpenAI(temperature=0, model="gpt-4o")
GPT4o_mini=ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
llm_stream = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",stream_usage=True,streaming=True)
llm_date = ChatOpenAI(temperature=0.3, model="gpt-4o-2024-05-13")


llm_screener = ChatOpenAI(temperature = 0.5 ,model ='gpt-4o-mini')

#POSTGRES