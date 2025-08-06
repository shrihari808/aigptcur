# ======================================================================
# DEPRECATED FILE
# ----------------------------------------------------------------------
# This file is deprecated and should not be used.
# Its presence is causing unexpected calls to the Bing Search API.
# The active implementation uses brave_news.py.
#
# If you see an error originating from this file, it means an old,
# cached version is being run or there is a rogue import statement
# in the application that needs to be removed.
#
# To fix:
# 1. Find the part of the code that imports from 'api.news_rag.bing_news'.
# 2. Replace it with an import from 'api.news_rag.brave_news'.
# 3. Delete the __pycache__ directory in this folder.
# 4. Restart the application server.
# ======================================================================

raise ImportError(
    "CRITICAL ERROR: The deprecated 'bing_news.py' module was imported. "
    "Please find the rogue import and replace it with 'brave_news.py'."
)






# import os
# import json
# import asyncio
# import requests
# import re
# from urllib.parse import urlparse
# import asyncpg
# import pandas as pd
# from openai import OpenAI
# from azure.cognitiveservices.search.websearch import WebSearchClient
# from azure.cognitiveservices.search.websearch.models import SafeSearch
# from msrest.authentication import CognitiveServicesCredentials
# from langchain_openai import ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain.prompts import (
#     PromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
# )
# from langchain.schema.runnable import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from pinecone import Pinecone as PineconeClient, ServerlessSpec
# from langchain.docstore.document import Document
# import psycopg2
# import pandas as pd
# from psycopg2 import sql


# from dotenv import load_dotenv
# load_dotenv()

# pg_ip=os.getenv('PG_IP_ADDRESS')
# psql_url=os.getenv('DATABASE_URL')

# # bing_api_key = os.getenv('BING_API_KEY')
# # Function to fetch news from Bing News Search API
# def fetch_news(query):
#     # --- DISABLED FUNCTION ---
#     # This function is deprecated and should not be called.
#     # It is intentionally left to raise an error to identify rogue calls.
#     print("ERROR: Deprecated function 'fetch_news' in bing_news.py was called.")
#     raise NotImplementedError("The Bing News search function 'fetch_news' is deprecated. Use the Brave News implementation.")
    

# def fetch_search(query):
#     # --- DISABLED FUNCTION ---
#     # This function is deprecated and should not be called.
#     # It is intentionally left to raise an error to identify rogue calls.
#     print("ERROR: Deprecated function 'fetch_search' in bing_news.py was called.")
#     raise NotImplementedError("The Bing News search function 'fetch_search' is deprecated. Use the Brave News implementation.")


# domain_list = [
#     "moneycontrol.com",
#     "economictimes.indiatimes.com",
#     "bloombergquint.com",
#     "business-standard.com",
#     "livemint.com",
#     "ndtv.com/business",
#     "cnbctv18.com",
#     "in.reuters.com",
#     "zeebiz.com",
#     "indiainfoline.com",
#     "financialexpress.com",
#     "thehindubusinessline.com",
#     "in.investing.com",
#     "money.rediff.com",
#     "businesstoday.in",
#     "marketmojo.com",
#     "moneylife.in",
#     "capitalmarket.com",
#     "nseindia.com",
#     "bseindia.com",
#     "indiatoday.in/business",
#     "economictimes.com",
#     "tradingeconomics.com/india",
#     "indiabudget.gov.in",
#     "businessworld.in",
#     "marketsmojo.com",
#     "dalalstreetinvestmentjournal.com",
#     "equitymaster.com",
#     "smartsavingadvice.com",
#     "goodreturns.in",
#     "stockmarket360.in",
#     "sharekhan.com",
#     "kotaksecurities.com",
#     "angelone.in",
#     "hdfcsec.com",
#     "motilaloswal.com",
#     "nirmalbang.com",
#     "karvyonline.com",
#     "reliancesmartmoney.com",
#     "yesbank.in",
#     "sbi.co.in",
#     "axisbank.com",
#     "icicidirect.com",
#     "edelweiss.in",
#     "5paisa.com",
#     "geojit.com",
#     "indiabulls.com",
#     "paisa.com",
#     "venturasecurities.com",
#     "bonanzaonline.com"
# ]

# def url_in_domain_list(url, domain_list):
#     # Parse the URL to get the domain
#     parsed_url = urlparse(url)
#     domain = parsed_url.netloc
    
#     # Check if the domain is in the list
#     if any(domain.endswith(d) for d in domain_list):
#         return True
#     else:
#         return None
    
# def process_search(search_results):
#     # This function depends on a disabled function and is therefore also deprecated.
#     return None, None


# def process_search_results(search_results):
#     # This function depends on a disabled function and is therefore also deprecated.
#     return None, None

# def insert_post1(db):
#     # This function is related to processing Bing results and is left as-is,
#     # but it will not be called if the search functions are disabled.
#     pass

# async def insert_post(df: pd.DataFrame):
#     # This function is related to processing Bing results and is left as-is,
#     # but it will not be called if the search functions are disabled.
#     pass

# # Function to initialize Pinecone
# def initialize_pinecone1():
#     # This function is related to processing Bing results and is left as-is,
#     # but it will not be called if the search functions are disabled.
#     pass

# # Function to get existing IDs (URLs) from Pinecone
# def data_into_pinecone1(df):
#     # This function is related to processing Bing results and is left as-is,
#     # but it will not be called if the search functions are disabled.
#     pass
    
# # Main function to get stock news summary
# def get_bing_results(query):
#     # --- DISABLED FUNCTION ---
#     print("ERROR: Deprecated function 'get_bing_results' in bing_news.py was called.")
#     raise NotImplementedError("The 'get_bing_results' function is deprecated. Use 'get_brave_results' instead.")


# async def initialize_pinecone():
#     pass

# async def process_row(row):
#     pass

# async def data_into_pinecone(df):
#     pass
