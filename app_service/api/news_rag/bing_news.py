import os
import json
import asyncio
import requests
import re
from urllib.parse import urlparse
import asyncpg
import pandas as pd
from openai import OpenAI
from azure.cognitiveservices.search.websearch import WebSearchClient
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.docstore.document import Document
import psycopg2
import pandas as pd
from psycopg2 import sql


from dotenv import load_dotenv
load_dotenv()

pg_ip=os.getenv('PG_IP_ADDRESS')
psql_url=os.getenv('DATABASE_URL')

# bing_api_key = os.getenv('BING_API_KEY')
# Function to fetch news from Bing News Search API
def fetch_news(query):
    bing_api_key = os.getenv('BING_API_KEY')
    subscription_key = bing_api_key
    endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    params = {
        "count": "10",
        "cc":'IND',
        "freshness": "Month",
        "q": query,
        "textDecorations": True,
        "mkt": "en-IN",
        "responseFilter": "News",
        "sortBy": "Date",
    }
    try:
        response = requests.get(url=endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        if 'value' in search_results and search_results['value']:
            return search_results
        else:
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")

    return None


def fetch_search(query):
    bing_api_key = os.getenv('BING_API_KEY')
    subscription_key = bing_api_key
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    params = {
    "count": "10",
    "cc":'IND',
    "freshness": "Month",
    "q": query,
    #"textDecorations": True,
    "mkt": "en-IN",
    #"responseFilter": "News",
    "sortBy": "Date",
    }
        


    try:
        response = requests.get(url=endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        if search_results['webPages']['value']:
            return search_results
        else:
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")

    return None


domain_list = [
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "bloombergquint.com",
    "business-standard.com",
    "livemint.com",
    "ndtv.com/business",
    "cnbctv18.com",
    "in.reuters.com",
    "zeebiz.com",
    "indiainfoline.com",
    "financialexpress.com",
    "thehindubusinessline.com",
    "in.investing.com",
    "money.rediff.com",
    "businesstoday.in",
    "marketmojo.com",
    "moneylife.in",
    "capitalmarket.com",
    "nseindia.com",
    "bseindia.com",
    "indiatoday.in/business",
    "economictimes.com",
    "tradingeconomics.com/india",
    "indiabudget.gov.in",
    "businessworld.in",
    "marketsmojo.com",
    "dalalstreetinvestmentjournal.com",
    "equitymaster.com",
    "smartsavingadvice.com",
    "goodreturns.in",
    "stockmarket360.in",
    "sharekhan.com",
    "kotaksecurities.com",
    "angelone.in",
    "hdfcsec.com",
    "motilaloswal.com",
    "nirmalbang.com",
    "karvyonline.com",
    "reliancesmartmoney.com",
    "yesbank.in",
    "sbi.co.in",
    "axisbank.com",
    "icicidirect.com",
    "edelweiss.in",
    "5paisa.com",
    "geojit.com",
    "indiabulls.com",
    "paisa.com",
    "venturasecurities.com",
    "bonanzaonline.com"
]

def url_in_domain_list(url, domain_list):
    # Parse the URL to get the domain
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the list
    if any(domain.endswith(d) for d in domain_list):
        return True
    else:
        return None
    
def process_search(search_results):
    # Safely get the 'value' key and check if it's not None and not empty
    values = search_results['webPages']['value']
    #print(len(values))
    
    if values:
        news = []
        for item in search_results['webPages']['value']:
            #print(item)
            if url_in_domain_list(item['url'],domain_list):
                news.append({
                        "title": item.get("name", None),
                        "source_url": item.get("url", None),
                        "image_url": None,  # Explicitly set to None
                        "description": item.get("snippet", None),
                        "heading": item.get('siteName', None),
                        # "image": item.get("image", {}).get("thumbnail", {}).get("contentUrl", None),
                        "source_date": item.get("datePublished", None),
                        "date_published": item.get("datePublished", "abc"),
                    })
            else:
                pass
        if len(news):
        
            # Create a DataFrame from the news list
            df = pd.DataFrame(news)
            
            # Clean up the 'title' and 'description' columns
            #df['title'] = df['title'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
            #df['description'] = df['description'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))    
            # Convert 'date_published' to a specific format and type
            #df["date_published"] = pd.to_datetime(df["date_published"]).dt.strftime('%Y%m%d').astype(int)
            
            # Drop rows with any NaN values
            #df.dropna(inplace=True)
            
            #print(df.columns)
            #print(df[['source_date', 'date_published']])
            filtered_articles = [{"source_url": article["source_url"], "title": article["title"], "description": article["description"]} for article in news]
            return filtered_articles,df
        else:
            return None,None
    else:
        return None,None


def process_search_results(search_results):
    # Safely get the 'value' key and check if it's not None and not empty
    values = search_results.get('value', [])
    #print(len(values))
    
    if values:
        news = []
        for item in values:
            #print(item)
            if url_in_domain_list(item['url'],domain_list):
                provider_info = item['provider'][0]
                # Check if image is provided in the main image section first
                image_url = item.get('image', {}).get('thumbnail', {}).get('contentUrl')
                # If no image is found in the main image section, check in the provider section
                if not image_url:
                    image_url = provider_info.get('image', {}).get('thumbnail', {}).get('contentUrl', None)
            # print(item['url'])
                news.append({
                    "source_url": item["url"],
                    "image_url": image_url,
                    "heading": provider_info['name'],  
                    "title": item["name"],
                    "description": item["description"],
                    "source_date":item['datePublished'],
                    "date_published": item["datePublished"],
                })
            else:
                pass
        if len(news):
        
            # Create a DataFrame from the news list
            df = pd.DataFrame(news)
            
            # Clean up the 'title' and 'description' columns
            df['title'] = df['title'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
            df['description'] = df['description'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))    
            # Convert 'date_published' to a specific format and type
            df["date_published"] = pd.to_datetime(df["date_published"]).dt.strftime('%Y%m%d').astype(int)
            
            # Drop rows with any NaN values
            df.dropna(inplace=True)
            
            #print(df.columns)
            #print(df[['source_date', 'date_published']])
            filtered_articles = [{"source_url": article["source_url"], "title": article["title"], "description": article["description"]} for article in news]
            return filtered_articles,df
        else:
            return None,None
    else:
        return None,None

def insert_post1(db):
# Assuming you have a DataFrame named df
    df = db

    # Database connection details
    db_url = psql_url

    try:
        # Connect to the database
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Extract values from the row
            #print(row)
            source_url = row['source_url']
            image_url = row['image_url']
            heading = row['heading']
            title = row['title']
            description = row['description']
            source_date=row['source_date']
            
            # # Check if the row already exists
            cur.execute("SELECT COUNT(1) FROM source_data WHERE source_url = %s", (source_url,))
            exists = cur.fetchone()[0]
            
            if exists:
                #print(f"Row with source_url {source_url} already exists, skipping...")
                continue
            
            #Prepare SQL query to insert data into the table
            insert_query = sql.SQL("""
                INSERT INTO source_data (source_url, image_url, heading, title, description,source_date)
                VALUES (%s, %s, %s, %s, %s,%s)
            """)
            # insert_query = sql.SQL("""
            #     INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
            #     VALUES (%s, %s, %s, %s, %s, %s)
            #     ON CONFLICT (source_url) DO NOTHING
            # """)
            
            # Execute the query with the extracted values
            cur.execute(insert_query, (source_url, image_url, heading, title, description,source_date))
        
        # Commit the transaction
        conn.commit()
       # print("Data inserted successfully into PostgreSQL")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the cursor and connection
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
  


async def insert_post(df: pd.DataFrame):
    db_url = psql_url
    #conn = None
    
    try:
        conn = await asyncpg.connect(db_url)
        
        for index, row in df.iterrows():
            source_url = row['source_url']
            image_url = row['image_url']
            heading = row['heading']
            title = row['title']
            description = row['description']
            source_date = row['source_date']
            
            # Check if the row already exists
            exists = await conn.fetchval("SELECT COUNT(1) FROM source_data WHERE source_url = $1", source_url)
            
            if exists:
                print(f"Row with source_url {source_url} already exists, skipping...")
                continue
            
            # Prepare SQL query to insert data into the table
            insert_query = """
                INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            # Execute the query with the extracted values
            await conn.execute(insert_query, source_url, image_url, heading, title, description, source_date)
        
        print("Data inserted successfully into PostgreSQL")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await conn.close()

# Function to initialize Pinecone
def initialize_pinecone1():
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    # pc.delete_index(index_name) 
    index_name = "bing-news"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )
    return pc, index_name

# Function to get existing IDs (URLs) from Pinecone
def data_into_pinecone1(df):
    _,index_name=initialize_pinecone()
    embeddings = OpenAIEmbeddings()
    document=[]
    ids = []
    for _,row in df.iterrows():
        combined_text = f'''{row["description"]}"'''
        document.append(Document(ids=row["source_url"],page_content=combined_text,metadata={"url":row["source_url"],"date":row["date_published"]}))
        url=row['source_url']
        cleaned_url = re.sub(r'[^a-zA-Z0-9]', '', url).lower()
        #print(cleaned_url)
        ids.append(cleaned_url)
    PineconeVectorStore.from_documents(documents=document, embedding=embeddings, index_name=index_name, namespace="bing", ids=ids)
    return "inserted"   
    
# Main function to get stock news summary
def get_bing_results(query):
    #res = fetch_news(query)
    res = fetch_search(query)

    #print(res)
    if res:
        #arts,df=process_search_results(res)
        arts,df=process_search(res)

        # print(arts)
        # print(df)
        if df is not None and not df.empty:

            #return df
            #insert_post(df)
            # pc, index_name = initialize_pinecone()
            # vectorstore = data_into_pinecone(df, index_name)
            return arts,df
        else:
            print("no df")
            return None,None
    else:
        return None,None
    # if df is not None and not df.empty:
    #     insert_post(df)
    #     pc, index_name = initialize_pinecone()
    #     vectorstore = data_into_pinecone(df, index_name)
    #     return vectorstore
    # else:
    #     print("no df")
    #     return None
    # insert_post(df)
    # pc, index_name = initialize_pinecone()
    # vectorstore = data_into_pinecone(df, index_name)
    # #review_chain = get_review_chain(vectorstore)
    # #response = review_chain.invoke(query)
        


# user_query = "which company bags orders today "
# response = get_stock_news_summary(user_query)
# print(response)
async def initialize_pinecone():
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "bing-news"
    if index_name not in (pc.list_indexes()).names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )
    return pc, index_name

async def process_row(row):
    combined_text = f'''{row["description"]}"'''
    document = Document(
        ids=row["source_url"],
        page_content=combined_text,
        metadata={"url": row["source_url"], "date": row["date_published"]}
    )
    url = row['source_url']
    cleaned_url = re.sub(r'[^a-zA-Z0-9]', '', url).lower()
    return document, cleaned_url

async def data_into_pinecone(df):
    pc, index_name = await initialize_pinecone()
    embeddings = OpenAIEmbeddings()
    documents = []
    ids = []

    tasks = [process_row(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    for document, cleaned_url in results:
        documents.append(document)
        ids.append(cleaned_url)

    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        namespace="bing",
        ids=ids
    )

    return "Inserted!"


# news=get_bing_results('latest news on relinace')
# print(news)