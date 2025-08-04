import aiohttp
import asyncio
import json
import os
import re
import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import spacy
import google.generativeai as genai
from fastapi import FastAPI, APIRouter
import uvicorn
import pandas as pd
from fuzzywuzzy import process
from typing import List
from datetime import datetime, timedelta
from api.serp_api import get_news
#from serp_api import get_news
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
import requests
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from starlette.status import HTTP_403_FORBIDDEN
from config import GPT4o_mini
router = APIRouter()
token = os.getenv('CMOTS_BEARER_TOKEN')

AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )
    

load_dotenv(override=True)

# Summary prompt
summary_template = '''  You are a financial news expert ,summarize the given news articles in 40 words of particular stock . Write short
 and crisp summary. Answer with a summary.
 {news}

'''
summary_prompt = PromptTemplate(template=summary_template, input_variables=["news"])
#llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini", max_tokens=1000)
llm_chain_summary = LLMChain(prompt=summary_prompt, llm=GPT4o_mini)

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
bearer_token = os.getenv("CMOTS_BEARER_TOKEN")

def find_stock_code1(stock_name):
    df = pd.read_csv("csv_data/company_codes2.csv")
    company_names = df["Company Name"].tolist()
    company_codes = df["Company Code"].tolist()
    threshold = 90
    match = process.extractOne(stock_name, company_names)
    if match and match[1] >= threshold:
        idx = company_names.index(match[0])
        return company_codes[idx]
    else:
        return None

async def get_data_from_url(session, url):
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                today = datetime.now().strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
                today_articles = [article for article in data["data"] if today in article["date"]]
                if not today_articles:
                    yesterday = (datetime.now() - timedelta(days=1)).strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
                    today_articles = [article for article in data["data"] if yesterday in article["date"]]
                if not today_articles:
                    day_before_yesterday = (datetime.now() - timedelta(days=2)).strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
                    today_articles = [article for article in data["data"] if day_before_yesterday in article["date"]]
                if not today_articles:
                    three_days_ago = (datetime.now() - timedelta(days=3)).strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
                    today_articles = [article for article in data["data"] if three_days_ago in article["date"]]
                if today_articles:
                    for article in today_articles:
                        clean_text = re.sub(r'<.*?>', '', article["arttext"])
                        article["arttext"] = clean_text
                    nlp = spacy.load("en_core_web_sm")
                    nlp_company_names = []
                    for article in today_articles:
                        doc = nlp(article["arttext"])
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                if not any(token.ent_type_ in ["DATE", "CARDINAL", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "TIME"] for token in ent):
                                    company_name = re.sub(r'[^\w\s]', '', ent.text)
                                    nlp_company_names.append(company_name.strip())
                    nlp_company_names = list(set(nlp_company_names))
                    return today_articles, len(nlp_company_names)
                else:
                    print("No articles found for today's date.")
                    return [], 0
            else:
                print("Failed to fetch data. Status code:", response.status)
                return [], 0
    except aiohttp.ClientError as e:
        print("Error fetching data:", e)
        return [], 0

def get_company_name(cleaned_articles, nlp_company_names_length):
    classification_prompt = [
        f"""
        You are a financial and stock expert tasked with analyzing news articles for companies. Follow these guidelines to extract relevant information:
        1. Company Extraction:
        - EXTRACT ALL COMPANY NAMES WITHOUT ANY BRACKETS (only COMPANIES, no sector names or any other generalized entities) FROM THE FOLLOWING ARTICLE TEXT: {cleaned_articles}.
        2. Reason Extraction:
        - Using the article data given in the "arttext" variable, give a DETAILED AND LONG reason why the company is in the news. If there is no reason, output "None"
        3. Stock/Volume Impact:
        - Analyze the effect of the news on the company's stock.
        - If the news affects the stock, identify the reason for the impact.
        - If the news doesn't affect the stock but affects the trading volume, specify the relevant details.
        4. Stock Impact Indicators:
        - Identify stock impact indicators such as "bags order," "launches," "gains," etc., for stock increase.
        - Identify stock impact indicators such as "slumps," "crashed," "decline," etc., for stock decrease.
        - Identify volume impact indicators such as "volumes spurt," "shares traded," "record high volume," etc., for volume increase.
        - Identify volume impact indicators such as "volumes dry up," "low volume," "record low volume," etc., for volume decrease.
        DO NOT PROVIDE ANY OFFENSIVE OR HARMFUL CONTENT IN THE RESPONSE.
        Provide output in STRICTLY the following format:
        ```
        1. Company: <exact_company_name>
            - Reason: <summary_of_arttext>
            - Stock/Volume Impact: <stock_or_volume_impact> 
        ```
        and so on until all companies mentioned in the given data are covered i.e. 70 companies at maximum.
        """
    ]
    response = model.generate_content(classification_prompt)
    company_pattern = r"Company: (.*?)(?:\n|$)"
    reason_pattern = r"Reason: (.*?)(?:\n|$)"
    stock_relevance_pattern = r"Stock/Volume Impact: (.*?)(?:\n|$)"
    try:
        company_names = re.findall(company_pattern, response.text)
        reasons = re.findall(reason_pattern, response.text)
        stock_relevance = re.findall(stock_relevance_pattern, response.text)
    except:
        company_names = []
        reasons = []
        stock_relevance = []
    return company_names, reasons, stock_relevance

async def get_ticker_symbol(session, company_name):
    formatted_company_name = "+".join(company_name.split())
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={formatted_company_name}&quotesCount=1&newsCount=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    }
    async with session.get(url, headers=headers) as response:
        data = await response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            ticker_symbol = data["quotes"][0]["symbol"]
            if "." in ticker_symbol:
                ticker_symbol = ticker_symbol.split(".")[0]
            return ticker_symbol
        else:
            words = company_name.split()
            if len(words) > 1:
                modified_sentence = ' '.join(words[:-1])
            formatted_company_name = "+".join(modified_sentence.split())
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={formatted_company_name}&quotesCount=1&newsCount=0"
            async with session.get(url, headers=headers) as response:
                data = await response.json()
                if "quotes" in data and len(data["quotes"]) > 0:
                    ticker_symbol = data["quotes"][0]["symbol"]
                    if "." in ticker_symbol:
                        ticker_symbol = ticker_symbol.split(".")[0]
                    return ticker_symbol
                else:
                    tic = find_stock_ticker(company_name)
                    return tic

def find_stock_ticker1(input_string):
    df = pd.read_csv("csv_data/EQUITY_L.csv")
    company_names = df["NAME OF COMPANY"].tolist()
    company_codes = df["SYMBOL"].tolist()
    threshold = 90
    match = process.extractOne(input_string, company_names)
    if match and match[1] >= threshold:
        idx = company_names.index(match[0])
        return company_codes[idx]
    else:
        return None


async def find_stock_code(stock_name):
    def load_data():
        return pd.read_csv("csv_data/6000stocks.csv")
    
    # Offload the CSV reading to a separate thread
    df = await asyncio.to_thread(load_data)
    # Extract the company names and codes from the DataFrame
    company_names = df["Company Name"].tolist()
    #company_codes = df["Company Code"].tolist()
    company_symbols =df['co_symbol'].tolist()
    
    threshold=90
    # Use fuzzy matching to find the closest match to the input stock name
    match = process.extractOne(stock_name, company_names)
    match1= process.extractOne(stock_name, company_symbols)

   
    if match and match[1] >= threshold:
        idx = company_names.index(match[0])
        #print(company_codes[idx])
        return company_symbols[idx]
    
    # Check if match in company symbols meets the threshold
    elif match1 and match1[1] >= threshold:
        idx = company_symbols.index(match1[0])
        #print(company_symbols[idx])
        return company_symbols[idx]

    else:
        return None
    

async def find_stock_ticker(input_string):
    def load_data():
        return pd.read_csv("csv_data/EQUITY_L.csv")
    
    # Offload the CSV reading to a separate thread
    df = await asyncio.to_thread(load_data)
    company_names = df["NAME OF COMPANY"].tolist()
    company_codes = df["SYMBOL"].tolist()
    threshold = 90
    match = process.extractOne(input_string, company_names)
    if match and match[1] >= threshold:
        idx = company_names.index(match[0])
        return company_codes[idx]
    else:
        return None
    
def sort_company_info(company_info):
    def extract_percentage_change(stock_relevance):
        match = re.search(r'(-?\d+(?:\.\d+)?)%', stock_relevance)
        if match:
            return float(match.group(1))
        return 0
    sorted_company_info = sorted(company_info, key=lambda x: abs(extract_percentage_change(x["stock_relevance"])), reverse=True)
    positive_changes = [company for company in sorted_company_info if extract_percentage_change(company["stock_relevance"]) >= 3]
    negative_changes = [company for company in sorted_company_info if extract_percentage_change(company["stock_relevance"]) <= -3]
    neutral_changes = [company for company in sorted_company_info if abs(extract_percentage_change(company["stock_relevance"])) < 3]
    sorted_company_info = positive_changes + neutral_changes + negative_changes
    return sorted_company_info

async def trending_stocks():
    print("running trending stocks")
    url1 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/hot-pursuit/30"
    url2 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/stock-alert/3"
    url3 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/corporate-news/30"
    url4 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/market-beat/30"
    urls = [url1, url2, url3, url4]
    combined_company_info = []
    async with aiohttp.ClientSession() as session:
        tasks = [get_data_from_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        for cleaned_articles, nlp_company_names_length in results:
            company_names, reasons, stock_relevance = get_company_name(cleaned_articles, nlp_company_names_length)
            company_stock_tickers = []
            #tasks = [get_ticker_symbol(session, name) for name in company_names]
            #tasks=[find_stock_ticker(name) for name in company_names]
            tasks=[find_stock_code(name) for name in company_names]
            tickers = await asyncio.gather(*tasks)
            company_stock_tickers.extend(tickers)
            company_info = []
            for name, ticker, reason, stock_relevance in zip(company_names, company_stock_tickers, reasons, stock_relevance):
                if ticker is None:
                    continue
                if reason == 'None':
                    news = get_stock_news(name)
                    company_dict = {
                        "name": name,
                        "stock_ticker": ticker,
                        "reason": news,
                        "stock_relevance": stock_relevance
                    }
                else:
                    company_dict = {
                        "name": name,
                        "stock_ticker": ticker,
                        "reason": reason,
                        "stock_relevance": stock_relevance
                    }
                company_info.append(company_dict)
            combined_company_info.extend(company_info)
    sorted_company_info = sort_company_info(combined_company_info)
    unique_entries = {}
    for entry in sorted_company_info:
        stock_ticker = entry["stock_ticker"]
        if entry["reason"] is None:
            continue
        if "dividend" not in entry["reason"].lower():
            if stock_ticker in unique_entries:
                try:
                    current_magnitude = float(unique_entries[stock_ticker]["stock_relevance"].split("%")[0])
                    new_magnitude = float(entry["stock_relevance"].split("%")[0])
                    if new_magnitude > current_magnitude:
                        unique_entries[stock_ticker] = entry
                except ValueError:
                    pass
            else:
                unique_entries[stock_ticker] = entry
    unique_data = list(unique_entries.values())
    return unique_data

def get_stock_news(company_name):
    company_code = find_stock_code(company_name)
    url = f"http://airrchipapis.cmots.com/api/NewDetails/{company_code}/-/10"
    token = os.getenv("CMOTS_BEARER_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        day = str(int(day))
        today_date = f"{month}/{day}/{year}"
        data = response.json()
        if data['data']:
            formatted_data_list = []
            for news_item in data['data']:
                iso_date = news_item['DATE'].split()[0]
                parsed_date = datetime.fromisoformat(iso_date)
                news_date = parsed_date.strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
                if news_date == today_date:
                    headline = news_item['heading']
                    formatted_data = {}
                    if len(news_item['arttext'].split()) > 60:
                        text = news_item['arttext']
                        summary = llm_chain_summary.run(text)
                        formatted_data['News'] = summary
                    else:
                        formatted_data['News'] = news_item['arttext']
                    formatted_data_list.append(formatted_data)
                    return formatted_data_list[0]['News']
        else:
            summary = get_news(company_name)
            return summary
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def get_db_connection(db_url):
    return psycopg2.connect(db_url)

def get_db_cursor(conn):
    return conn.cursor()

def store_into_db(result_json):
    print("result", result_json)
    pg_ip = os.getenv('PG_IP_ADDRESS')
    psql_url=os.getenv('DATABASE_URL')
    db_url = psql_url
    
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Update all existing entries to set isactive to false
            update_query = sql.SQL("""
                UPDATE trending_stocks
                SET \"isActive\" = 'False'
            """)
            cur.execute(update_query)
            
            # Prepare the insert query
            insert_query = sql.SQL("""
                INSERT INTO trending_stocks (name, stock_ticker, reason, stock_relevance)
                VALUES (%s, %s, %s, %s)
            """)

            # Iterate over the JSON data and insert each record
            for item in result_json:
                cur.execute(insert_query, (item['name'], item['stock_ticker'], item['reason'], item['stock_relevance']))

            # Commit the transaction
            conn.commit()
            print("Data inserted into trending_stocks table and isactive updated")

@router.get("/trending_stocks")
async def get_trend_stocks(ai_key_auth: str = Depends(authenticate_ai_key)):
    data = await trending_stocks()
    store_into_db(data)
    

# if __name__ == "__main__":
#     app = FastAPI()
#     app.include_router(router)
#     uvicorn.run(app, host="0.0.0.0", port=8000)
