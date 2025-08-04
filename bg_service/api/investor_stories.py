import requests
import asyncio
import json
import pytz
from psycopg2 import sql, extras
from typing import List
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from datetime import time
from datetime import datetime, timedelta
import pandas as pd
from fuzzywuzzy import process
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from starlette.status import HTTP_403_FORBIDDEN
from config import GPT4o_mini

load_dotenv()

router=APIRouter()




token = os.getenv('CMOTS_BEARER_TOKEN')
openai_api_key=os.getenv('OPENAI_API_KEY')
pg_ip=os.getenv('PG_IP_ADDRESS')
psql_url=os.getenv('DATABASE_URL')


def process_headline(headline):
    inputs = tokenizer(headline, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
    # Check if either the first or second index has a score greater than 0.9
    if predictions[0] > 0.9 or predictions[1] > 0.9 or predictions[2] > 0.9:
        result = 'impact'
    else:
        result = 'no impact'
    #print(predictions)
    #max_score_index = predictions.index(max(predictions))
    return result

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


summary_template = '''  You are a financial news expert ,summarize the given news article in 60 words . Answer with a summary.
 news article:{news}

'''
summary_prompt = PromptTemplate(template=summary_template, input_variables=["news"])
# llm_summary = OpenAI(temperature=0.6)
#llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini", max_tokens=1000)
llm_chain_summary = LLMChain(prompt=summary_prompt, llm=GPT4o_mini)

@contextmanager
def get_db_connection(db_url):
    conn = psycopg2.connect(db_url)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        conn.close()

@contextmanager
def get_db_cursor(conn):
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def store_into_db(result_json):
    db_url = psql_url
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Update all existing entries to set isactive to false
            update_query = sql.SQL("""
                UPDATE investor_stores_data
                SET \"isActive\" = 'False'
            """)
            cur.execute(update_query)
            
            # Prepare the insert query
            # insert_query = sql.SQL("""
            #     INSERT INTO investor_stores_data (data)
            #     VALUES (%s::jsonb)
            # """)

                        # Convert result_json to JSON string
            result_json_str = json.dumps(result_json)
            #print(result_json_str)

            # Deactivate all current active entries
            #cur.execute(deactivate_query)

            # Insert the new data with isActive set to TRUE
            #cur.execute(insert_query, {result_json_str})
            cur.execute(f''' 
                    INSERT INTO 
                        investor_stores_data(data,"isActive")  
                    VALUES 
                        ('{result_json_str}','True') 
                ''')

            # Commit the transaction
            conn.commit()
            #is_active = True 
            # Execute the insert query with the JSON data
            #cur.execute(insert_query, (result_json,))
            #cur.execute(insert_query, (result_json))
            print("Data inserted into DB and isactive updated")
     
        

def get_summary(sno,type,text):
    db_url = psql_url
    try:
        with get_db_connection(db_url) as conn:
            with get_db_cursor(conn) as cur:
                # Check if the row exists
                cur.execute("SELECT summary FROM investor_stores WHERE sno = %s", (sno,))
                row = cur.fetchone()
                
                if row is not None:
                    exists = row[0]
                    #print("exis:",exists)
                    return exists  # Return the summary if it exists
                else:
                    # Generate summary
                    summary = llm_chain_summary.run(text)
                    cleaned_summary = summary.replace("'", "")
                    #print("sum:",summary)
                    
                    # Prepare SQL query to insert data into the table
                    insert_query = """
                        INSERT INTO investor_stores (sno, type, summary)
                        VALUES (%s, %s, %s)
                    """
                    
                    # Execute the query with the extracted values
                    cur.execute(insert_query, (sno, type, cleaned_summary))

                    # Commit the transaction
                    conn.commit()
                    return cleaned_summary

    except Exception as e:
        return f"Error: {e}"



keywords = [
    "Earnings", "Revenue", "Profits", "Sales", "Growth", "Acquisition", "Merger",
    "Partnership", "IPO", "Dividend", "Stock split", "Layoffs", "Lawsuit",
    "Regulation", "Analyst upgrade/downgrade", "Guidance", "CEO", "CFO",
    "Product launch", "FDA approval", "Economic indicators", "Interest rates",
    "Federal Reserve", "Trade tensions", "Tariffs", "Geopolitical events",
    "Market volatility", "Supply chain disruptions", "Natural disasters",
    "Technological advancements", "Consumer sentiment", "Quarterly report",
    "Financial results", "Market performance", "Investor sentiment",
    "Market sentiment", "Stock market", "Price target", "Valuation",
    "Revenue forecast", "Earnings forecast", "Profit forecast",
    "Revenue guidance", "Earnings guidance", "Profit guidance",
    "Market trend", "Market outlook","Bullish", "Bearish", "Volatility", "Trend", "Sentiment", "Rally",
    "Slump", "Correction", "Inflation", "Deflation", "Unemployment", "Jobless claims", "Consumer spending",
    "Consumer confidence", "Housing market", "Housing starts", "Retail sales", "Manufacturing index", "Consumer price index",
    "Producer price index", "Purchasing managers' index", "Trade balance", "Current account", "Gross domestic product", "Growth rate",
    "Productivity", "Inflation rate", "Interest rate hike", "Interest rate cut", "Monetary policy", "Fiscal policy", "Government spending",
    "Tax policy", "Debt ceiling", "Credit rating", "Central bank", "Treasury yields", "Bond market", "Commodity prices", "Oil prices", "Gold prices",
    "Currency exchange rates", "Exchange rate", "Foreign exchange", "Institutional investors", "Retail investors", "Hedge funds", "Mutual funds",
    "Exchange-traded funds", "High-frequency trading", "Algorithmic trading", "Short selling", "Liquidity", "Margin call", "Market manipulation",
    "Insider trading", "Pump and dump", "Market psychology", "Speculation", "Bubble", "Crash", "Recession", "Depression", "Global economic conditions",
    "high","spike"
]


def relevance(headline, keywords):
    count = 0
    for keyword in keywords:
        if fuzz.partial_ratio(headline.lower(), keyword.lower()) > 60:  # Adjust the threshold as needed
            count += 1
    return count


def find_stock_code(stock_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(r"csv_data\company_codes2.csv")

    # Extract the company names and codes from the DataFrame
    company_names = df["Company Name"].tolist()
    company_codes = df["Company Code"].tolist()

    threshold = 80
    # Use fuzzy matching to find the closest match to the input stock name
    match = process.extractOne(stock_name, company_names)

    if match and match[1] >= threshold:
        # Get the index of the closest match
        idx = company_names.index(match[0])
        # Return the corresponding stock code
        return company_codes[idx]
    else:
        return None


async def hot_news(keyword):
    #type='hot-pursuit'
    url = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/hot-pursuit/20"
    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse today's date
        ist = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist).strftime("%m/%d/%Y").lstrip("0")
        #today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        # Remove leading zeros from the day
        day = str(int(day))
        # Reconstruct the date string
        today_date = f"{month}/{day}/{year}"
        # print(today_date)

        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            count=0
            for news_item in data['data']:
                sno=news_item['sno']

                #print(count)
                if count >=5 :
                    break
                # Get the date from the news item
                news_date = news_item['date'].split()[0]
                # print(news_date)
                # Check if the news date matches today's date
                if news_date == today_date:
                    headline = news_item['heading'].replace("'", "")
                    # Check if the headline has nonzero similarity score
                    if relevance(headline, keyword) != 0 and process_headline(headline)=='impact':
                        formatted_data = {
                            "Headline": headline,
                            # "Caption": news_item['caption'],
                        }
                        # Check if the news text has more than 60 words
                        if len(news_item['arttext'].split()) > 60:
                            text = news_item['arttext']
                            sno = news_item['sno']
                            summary = get_summary(sno,'hot-pursuit',text)
                            formatted_data['News'] = summary
                        else:
                            formatted_data['News'] = news_item['arttext'].replace("'", "")
                        #formatted_data_list.append(formatted_data)

                        time_stamp= f"{news_item['date'].split()[0]} {news_item['time']}"
                        formatted_data['time_stamp']=time_stamp
                        formatted_data_list.append(formatted_data)
                        count += 1  # Increment count
            return formatted_data_list

        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)


async def corp_news(keyword):
    #type='corp_news'
    url = f"http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/corporate-news/20"
    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse today's date
        ist = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist).strftime("%m/%d/%Y").lstrip("0")
        #today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        # Remove leading zeros from the day
        day = str(int(day))
        # Reconstruct the date string
        today_date = f"{month}/{day}/{year}"
        # print(today_date)

        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            count=0
            for news_item in data['data']:
                if count >= 5:
                    break
                # Get the date from the news item
                news_date = news_item['date'].split()[0]
                # print(news_date)
                # Check if the news date matches today's date
                if news_date == today_date:
                    headline = news_item['heading'].replace("'", "")
                    # Check if the headline has nonzero similarity score
                    if relevance(headline, keyword) != 0 and process_headline(headline)=='impact':
                        formatted_data = {
                            "Headline": headline,
                            # "Caption": news_item['caption'],
                        }
                        # Check if the news text has more than 60 words
                        if len(news_item['arttext'].split()) > 60:
                            text = news_item['arttext']
                            sno=news_item['sno']
                            summary = get_summary(sno,'corp_news',text)
                            formatted_data['News'] = summary
                        else:
                            formatted_data['News'] = news_item['arttext'].replace("'", "")

                        time_stamp= f"{news_item['date'].split()[0]} {news_item['time']}"
                        formatted_data['time_stamp']=time_stamp
                        formatted_data_list.append(formatted_data)
                        count += 1
                        
            return formatted_data_list

        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)


async def economy_news(keyword):
    #type='eco_news'
    url = f"http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/economy-news/20"
    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse today's date
        ist = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist).strftime("%m/%d/%Y").lstrip("0")
        #today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        # Remove leading zeros from the day
        day = str(int(day))
        # Reconstruct the date string
        today_date = f"{month}/{day}/{year}"
        # print(today_date)

        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            count = 0 
            for news_item in data['data']:
                if count>=5:
                    break
                # Get the date from the news item
                news_date = news_item['date'].split()[0]
                # print(news_date)
                # Check if the news date matches today's date
                if news_date == today_date:
                    headline = news_item['heading'].replace("'", "")
                    # Check if the headline has nonzero similarity score
                    if relevance(headline, keyword) != 0 and process_headline(headline)=='impact':
                        formatted_data = {
                            "Headline": headline,
                            # "Caption": news_item['caption'],
                        }
                        # Check if the news text has more than 60 words
                        if len(news_item['arttext'].split()) > 60:
                            text = news_item['arttext']
                            sno=news_item['sno']
                            summary = get_summary(sno,'eco_news',text)
                            formatted_data['News'] = summary
                        else:
                            formatted_data['News'] = news_item['arttext'].replace("'", "")
                        time_stamp= f"{news_item['date'].split()[0]} {news_item['time']}"
                        formatted_data['time_stamp']=time_stamp
                        formatted_data_list.append(formatted_data)
                        count += 1
                        
            return formatted_data_list

        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)


async def corp_results(keyword):
    #type='corp-res'
    url = f"http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/corporate-results/20"
    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse today's date
        ist = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist).strftime("%m/%d/%Y").lstrip("0")
        #today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        # Remove leading zeros from the day
        day = str(int(day))
        # Reconstruct the date string
        today_date = f"{month}/{day}/{year}"
        # print(today_date)

        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            count=0
            for news_item in data['data']:
                if count>=5:
                    break
                # Get the date from the news item
                news_date = news_item['date'].split()[0]
                # print(news_date)
                # Check if the news date matches today's date
                if news_date == today_date:
                    headline = news_item['heading'].replace("'", "")
                    # Check if the headline has nonzero similarity score
                    if relevance(headline, keyword) != 0 and process_headline(headline)=='impact':
                        formatted_data = {
                            "Headline": headline,
                            # "Caption": news_item['caption'],
                        }
                        # Check if the news text has more than 60 words
                        if len(news_item['arttext'].split()) > 60:
                            text = news_item['arttext']
                            sno=news_item['sno']
                            summary = get_summary(sno,'corp-res',text)
                            formatted_data['News'] = summary
                        else:
                            formatted_data['News'] = news_item['arttext'].replace("'", "")
                        time_stamp= f"{news_item['date'].split()[0]} {news_item['time']}"
                        formatted_data['time_stamp']=time_stamp
                        formatted_data_list.append(formatted_data)
                        count += 1
                        
            return formatted_data_list

        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)


async def market_news(keyword):
    #type='market_news'
    url = f"http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/market-beat/20"
    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse today's date
        ist = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist).strftime("%m/%d/%Y").lstrip("0")
        #today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        # Remove leading zeros from the day
        day = str(int(day))
        # Reconstruct the date string
        today_date = f"{month}/{day}/{year}"
        # print(today_date)

        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            count=0
            for news_item in data['data']:
                if count>=5:
                    break
                # Get the date from the news item
                news_date = news_item['date'].split()[0]
                # print(news_date)
                # Check if the news date matches today's date
                if news_date == today_date:
                    headline = news_item['heading'].replace("'", "")
                    # Check if the headline has nonzero similarity score
                    if relevance(headline, keyword) != 0 and process_headline(headline)=='impact':
                        formatted_data = {
                            "Headline": headline,
                            # "Caption": news_item['caption'],
                        }
                        # Check if the news text has more than 60 words
                        if len(news_item['arttext'].split()) > 60:
                            text = news_item['arttext']
                            sno=news_item['sno']
                            summary = get_summary(sno,'market_news',text)
                            formatted_data['News'] = summary
                        else:
                            formatted_data['News'] = news_item['arttext'].replace("'", "")
                        time_stamp= f"{news_item['date'].split()[0]} {news_item['time']}"
                        formatted_data['time_stamp']=time_stamp
                        formatted_data_list.append(formatted_data)
                        count += 1
            return formatted_data_list

        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)


async def session_news():
    pre_s = f'http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/pre-session/5'
    mid_s = f'http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/mid-session/5'
    end_s = f'http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/end-session/5'

    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).time()
    print(current_time)
    # current_time = datetime.now().time()
    # print(current_time)
    pre_session_end = time(9, 0)  # Pre-session ends at 9:00 AM
    end_session_start = time(16, 0)  # End-session starts at 4:00 PM

    if current_time < pre_session_end:
        url_to_use = pre_s
    elif current_time < end_session_start:
        url_to_use = mid_s
    else:
        url_to_use = end_s

    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url_to_use, headers=headers)
    if response.status_code == 200:
        # Parse today's date
        today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        # Remove leading zeros from the day
        day = str(int(day))
        # Reconstruct the date string
        today_date = f"{month}/{day}/{year}"
        # print(today_date)

        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            count=0
            for news_item in data['data']:
                if count>=5:
                    break
                # Get the date from the news item
                news_date = news_item['date'].split()[0]
                # print(news_date)
                # Check if the news date matches today's date
                if news_date == today_date:
                    headline = news_item['heading'].replace("'", "")
                    formatted_data = {
                        "Headline": headline,
                        # "Caption": news_item['caption'],
                    }
                    # Check if the news text has more than 60 words
                    if len(news_item['arttext'].split()) > 60:
                        text = news_item['arttext']
                        sno=news_item['sno']
                        summary =get_summary(sno,'session',text)
                        formatted_data['News'] = summary
                    else:
                        formatted_data['News'] = news_item['arttext'].replace("'", "")
                    time_stamp= f"{news_item['date'].split()[0]} {news_item['time']}"
                    formatted_data['time_stamp']=time_stamp
                    formatted_data_list.append(formatted_data)
                    count += 1
            return formatted_data_list
    # # Check if the request was successful (status code 200)
    # if response.status_code == 200:
    #     # Print the response content
    #     data = response.json()
    #     if data['data']:
    #         formatted_data_list = []
    #         if 'data' in data and isinstance(data['data'], list):
    #             up_data_list = data['data']

    #             for dil_data in reversed(up_data_list):
    #                 if isinstance(dil_data, dict):
    #                     formatted_data = {
    #                         "Headline": dil_data.get('heading'),
    #                         # "caption": dil_data['caption']
    #                     }
    #                     # Check if the news text has more than 60 words
    #                     if len(dil_data['arttext'].split()) > 60:
    #                         text =dil_data['arttext']
    #                         summary = llm_chain_summary.run(text)
    #                         formatted_data['News'] = summary
    #                     else:
    #                         formatted_data['News'] = dil_data['arttext']
    #                     time_stamp= f"{dil_data['date'].split()[0]} {dil_data['time']}"
    #                     formatted_data['time_stamp']=time_stamp
    #                     formatted_data_list.append(formatted_data)
                        
    #         return formatted_data_list
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)


async def stock_news(stock_name):
    all_formatted_data_list = []
    for st in stock_name:
        company_code = find_stock_code(st)
        url = f"http://airrchipapis.cmots.com/api/NewDetails/{company_code}/-/10"
        # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token

        # Set up the headers with the authorization token
        headers = {"Authorization": f"Bearer {token}"}

        # Make a GET request with headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse today's date
            today_date = datetime.now().strftime("%Y-%m-%d")
            yesterday_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")

            # Print the response content
            data = response.json()
            if data['data']:
                formatted_data_list = []
                for news_item in data['data']:
                    # Get the date from the news item
                    news_date = news_item['DATE'].split('T')[0]

                    # Check if the news date matches today's date or yesterday's date
                    if news_date in [today_date, yesterday_date]:
                        headline = news_item['heading'].replace("'", "")
                        # Check if the headline has nonzero similarity score
                        if relevance(headline, keywords) != 0 and process_headline(headline)=='impact':
                            formatted_data = {
                                "Headline": headline,
                                # "Caption": news_item['caption']
                            }

                            # Check if the news text has more than 60 words
                            num_words = len(news_item['arttext'].split())
                            if num_words > 60:
                                text = news_item['arttext']
                                sno=news_item['sno']
                                summary = get_summary(sno,'stock_news',text)
                                formatted_data['Summary'] = summary
                            else:
                                formatted_data['News'] = news_item['arttext'].replace("'", "")
                            iso_date = news_item['DATE'].split()[0]
                            parsed_date = datetime.fromisoformat(iso_date)
                            news_date = parsed_date.strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
                            time_stamp= f"{news_date} {news_item['time']}"
                            formatted_data['time_stamp']=time_stamp
                            formatted_data_list.append(formatted_data)
                            

                return formatted_data_list
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)


async def stock_alerts(stock_name):
    url = f"http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/stock-alert/5"
    # token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse today's date
        today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
        month, day, year = today_date.split('/')
        # Remove leading zeros from the day
        day = str(int(day))
        # Reconstruct the date string
        today_date = f"{month}/{day}/{year}"
        # print(today_date)

        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            for news_item in data['data']:
                # Get the date from the news item
                news_date = news_item['date'].split()[0]
                #print(news_date)
                # Check if the news date matches today's date
                if news_date == today_date:
                    headline = news_item['heading'].replace("'", "")
                    # Check if the headline has nonzero similarity score
                    if relevance(headline, stock_name) != 0 and process_headline(headline)=='impact':
                        formatted_data = {
                            "Headline": headline,
                            # "Caption": news_item['caption'],
                        }
                        # Check if the news text has more than 60 words
                        if len(news_item['arttext'].split()) > 60:
                            text = news_item['arttext']
                            sno=news_item['sno']
                            summary = get_summary(sno,'stock_alert',text)
                            formatted_data['News'] = summary
                        else:
                            formatted_data['News'] = news_item['arttext'].replace("'", "")
                        time_stamp= f"{news_item['date'].split()[0]} {news_item['time']}"
                        formatted_data['time_stamp']=time_stamp
                        formatted_data_list.append(formatted_data)
                        
            return formatted_data_list

        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)



# class StockRequest(BaseModel):
#     stock_names: List[str]




#@router.post("/investor_stories")
# @router.post("/investor_stories")
# async def process_data():
    

#     stocks = []
#     keywords_with_stocks = keywords

#     stock_new, stock_alert, hot_pursuit_news, corporate_news, economic_news, corporate_results_news, mark_news, session = await asyncio.gather(
#     stock_news(stocks),
#     stock_alerts(stocks),
#     hot_news(keywords_with_stocks),
#     corp_news(keywords_with_stocks),
#     economy_news(keywords_with_stocks),
#     corp_results(keywords_with_stocks),
#     market_news(keywords_with_stocks),
#     session_news(),
#     )
#     combined_stock_news = []
#     if stock_new:
#         combined_stock_news.extend(stock_new)
#     if stock_alert:
#         combined_stock_news.extend(stock_alert)
#     #combined_stock_news = stock_new + stock_alert


#     res_json= {
#     "watchlist_news": combined_stock_news,
#     "session_news": session,
#     "hot_pursuit_news": hot_pursuit_news,
#     "corporate_news": corporate_news,
#     "economy_news": economic_news,
#     "corporate_results_news": corporate_results_news,
#     "market_news": mark_news,
#     }

#     #return res_json
#     # print(type(res_json))
#     # print(res_json)

#     # result_json_str = json.dumps(res_json)
#     # print(result_json_str)
#     # print(type(result_json_str))

#     # #store_into_db(result_json_str)
#     store_into_db(res_json)

#     return "data inserted successfully"

AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )




@router.post("/investor_stories")
async def process_data(ai_key_auth: str = Depends(authenticate_ai_key)):
    stocks = []
    keywords_with_stocks = keywords

    # stock_new, stock_alert, hot_pursuit_news, corporate_news, economic_news, corporate_results_news, mark_news, session = await asyncio.gather(
    # stock_news(stocks),
    # stock_alerts(stocks),
    # hot_news(keywords_with_stocks),
    # corp_news(keywords_with_stocks),
    # economy_news(keywords_with_stocks),
    # corp_results(keywords_with_stocks),
    # market_news(keywords_with_stocks),
    # session_news(),
    # )
    hot_pursuit_news, corporate_news, economic_news, corporate_results_news, mark_news, session = await asyncio.gather(
    hot_news(keywords_with_stocks),
    corp_news(keywords_with_stocks),
    economy_news(keywords_with_stocks),
    corp_results(keywords_with_stocks),
    market_news(keywords_with_stocks),
    session_news(),
    )
    # combined_stock_news = []
    # if stock_new:
    #     combined_stock_news.extend(stock_new)
    # if stock_alert:
    #     combined_stock_news.extend(stock_alert)

    res_json = {
        #"watchlist_news": combined_stock_news,
        "session_news": session,
        "hot_pursuit_news": hot_pursuit_news,
        "corporate_news": corporate_news,
        "economy_news": economic_news,
        "corporate_results_news": corporate_results_news,
        "market_news": mark_news,
    }

    store_into_db(res_json)

    return JSONResponse(
        content={"message": "Data inserted successfully"},
        status_code=201
    )

# def lambda_handler(event, context):
#     # Handle Lambda event and context if necessary
#     asyncio.run(process_data())  # Run the async function synchronously in Lambda

#     return {
#         'statusCode': 200,
#         'body': json.dumps('Data inserted successfully')
#     }


# asyncio.run(process_data())
# sno='1523731'
# type='hot-pursuit'
# text='''The order, valued at Rs 1.87 crore, involves supplying main turbine spares for the NTPC Talcher plant. GE Power India will fulfill the order within 9.5 months. <P> GE Power India's expertise covers engineering, manufacturing, project management, and supplying products and equipment for power plants. They operate across the entire power plant lifecycle, from design and procurement to construction and servicing. <P> The company reported a consolidated net profit of Rs 25.94 crore in Q4 FY24 as against a net loss of Rs 129.7 crore in Q4 FY23. Revenue from operations rose by 13.43% year on year (YoY) to Rs 390.76 crore in the fourth quarter of FY24.  <P><p><b><i>Powered by</i> Capital Market - Live News</b></p>'''

# get_summary(sno,type,text)