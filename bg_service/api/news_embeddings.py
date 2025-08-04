import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import json
from pinecone import PodSpec, Pinecone as PineconeClient
import requests
import re
from langchain.docstore.document import Document
from datetime import datetime
from langchain_pinecone import Pinecone
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import psycopg2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import chromadb.utils.embedding_functions as embedding_functions
#from config import chroma_server_client
from langchain_chroma import Chroma
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from starlette.status import HTTP_403_FORBIDDEN

from config import chroma_server_client,vs,GPT4o_mini

# GPT4o_mini=ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )




router=APIRouter()
load_dotenv(override=True)

# # Access the Bearer token
bearer_token = os.getenv('CMOTS_BEARER_TOKEN')
openai_api_key=os.getenv('OPENAI_API_KEY')
Pinecone_api_key=os.getenv('PINECONE_API_KEY')
pg_ip=os.getenv('PG_IP_ADDRESS')
psql_url=os.getenv('DATABASE_URL')

# #index_name = "news"
# index_name = "newsrag11052024"
# demo_namespace='news'
# embeddings = OpenAIEmbeddings()

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# pc = PineconeClient(
#  api_key=Pinecone_api_key)


# index = pc.Index(index_name)


#chroma


# import chromadb
# from chromadb.config import Settings,DEFAULT_DATABASE,DEFAULT_TENANT
# import os
# from langchain_openai import ChatOpenAI



# chroma_username=os.getenv("CHROMA_USERNAME")
# chroma_password=os.getenv("CHROMA_PASSWORD")
# chroma_host=os.getenv("CHROMA_HOST")

# # print(chroma_host)
# chroma_server_client=client = chromadb.HttpClient(
#     host=chroma_host,
#     port=8000,
#     ssl=False,
#     headers=None,
#     settings=Settings(chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",chroma_client_auth_credentials=f"{chroma_username}:{chroma_password}", allow_reset=True),
#     tenant=DEFAULT_TENANT,
#     database=DEFAULT_DATABASE,
# )


# client=chroma_server_client
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key=os.getenv("OPENAI_API_KEY"),
#                 model_name="text-embedding-3-small"
#             )
# # collection = client.create_collection(name="cmots_news", embedding_function=openai_ef)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vs= Chroma(
#     client=client,
#     collection_name="cmots_news",
#     embedding_function=embeddings,)


def get_data_from_url(url,bearer_token):

    #token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzE5MzA2NTU2LCJleHAiOjE3MjAyNTY5NTYsImlhdCI6MTcxOTMwNjU1NiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.yARrqS-eeDnmc4QXrBUPv7qPR9e2GhBlsV8q6wQTMZM'
    
    headers = {"Authorization": f"Bearer {bearer_token}"}

    try:
        # Sending GET request with headers
        response = requests.get(url, headers=headers)

        # Checking if request was successful (status code 200)
        if response.status_code == 200:
            # Extracting JSON data
            data = response.json()
            
            articles = data["data"]
            
            today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
            month, day, year = today_date.split('/')
            # Remove leading zeros from the day
            day = str(int(day))
            # Reconstruct the date string
            today_date = f"{month}/{day}/{year}"
            #today_articles = [article for article in data["data"] if today_date in article["date"]]
            today_articles=articles
    
                    
            if today_articles:
                for article in today_articles:
                    art=article.get("arttext")
                    if art:
                        clean_text = re.sub(r'<.*?>', '', article["arttext"])
                        article["arttext"] = clean_text
                    else:
                        pass
            

                return today_articles
            else:
                print("No articles found for today's date.")
                return []
        else:
            print("Failed to fetch data. Status code:", response.status_code)

    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)


def get_articles():
    url1 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/hot-pursuit/50"
    url2 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/stock-alert/5"
    url3 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/corporate-news/50"
    url4 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/market-beat/50"
    url5 =" http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/ipo-news/50"
    url6 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/economy-news/50"
    url7 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/corporate-results/50"
    url8 ="http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/pre-session/10"
    url9 ="http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/mid-session/10"
    url10="http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/end-session/10"

    urls = [url1, url2, url3, url4, url5,url6,url7,url8,url9,url10]
    #urls=[url1]
    
    
    articles = []
    for url in urls:  
        print("Fetching data from URL:", url)
        cleaned_articles = get_data_from_url(url,bearer_token)
        print("Number of articles fetched:", len(cleaned_articles))
        articles.extend(cleaned_articles)
        
    # print(json.dumps(articles, indent = 4))
    return articles


def data(arts):
    # arts=get_articles()
    documents = []
    ids_new=[]
    existing_ids = set()  # Initialize an empty set
    # for ids in index.list(namespace='news'):
    #     existing_ids.update(ids) 
    for article in arts:
        # existing_ids = set()  # Initialize an empty set
        # for ids in index.list(namespace='newsrag'):
        #     existing_ids.update(ids) 
        if article["sno"] not in existing_ids:

            date_object = datetime.strptime(article["date"], "%m/%d/%Y %I:%M:%S %p")
            # Extracting only the date part
            date_only = date_object.date()
            date_without_dash = date_only.strftime("%Y%m%d")

            # Converting the date without dashes to an integer
            int_date = int(date_without_dash)
    #         text= f"""
    #                 "heading": {article["heading"]},
                    
    #                 "caption": {article["caption"]},
                    
    #                 "article_text": {article["arttext"]}
    #                 """,
            #combined_text = article["heading"] + " " + article["caption"] + " " + article["arttext"]
            #combined_text = f"heading: {article['heading']} caption: [{article['caption']}] text: [{article['arttext']}]"
            combined_text = f'''"News_type":{article['section_name']}: heading: {article['heading']} caption: {article['caption']} text: {article['arttext']}"'''
            metadatas={
                "date": int_date
            }
            #print(metadatas)
            id_art= article["sno"]
            print(id_art)
            
            
            documents.append(Document(ids=id_art, page_content=combined_text, metadata=metadatas))
            ids_new.append(id_art)
            #docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace=demo_namespace, ids=id_art)
            
            print("Article inserted into the database:", article["sno"])
        else:
                print("Article already exists in the database:", article["sno"])

    #docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace=demo_namespace, ids=ids)
    return documents,ids_new
    print("Data insertion completed.")


def get_data_from_sec_url(url,bearer_token):
    #token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzE2NDQ1NDM0LCJleHAiOjE3MTkzODMwMzQsImlhdCI6MTcxNjQ0NTQzNCwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.FS_3GZ4PzbnXepFT0wYJa0NdvY4mZoCua2Yvyj_lY50'
    
    headers = {"Authorization": f"Bearer {bearer_token}"}

    try:
        # Sending GET request with headers
        response = requests.get(url, headers=headers)

        # Checking if request was successful (status code 200)
        if response.status_code == 200:
            # Extracting JSON data
            data = response.json()
            
            articles = data["data"]
    
            if articles:
                for article in articles:
                    # Cleaning HTML tags from article text
                    art=article.get("arttext")
                    if art:
                        clean_text = re.sub(r'<.*?>', '', article["arttext"])
                        article["arttext"] = clean_text
                    else:
                        pass
                    
                return articles
            else:
                print("No articles found for today's date.")
                return []
        else:
            print("Failed to fetch data. Status code:", response.status_code)

    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)



def sector_articles():
    # sectors = [
    # {'sect_code': '00000005', 'sect_name': 'Automobile'},
    # {'sect_code': '00000006', 'sect_name': 'Banks'},
    # {'sect_code': '00000011', 'sect_name': 'Cement'},
    # {'sect_code': '00000019', 'sect_name': 'Crude Oil & Natural Gas'},
    # {'sect_code': '00000020', 'sect_name': 'Diamond, Gems and Jewellery'},
    # {'sect_code': '00000026', 'sect_name': 'Finance'},
    # {'sect_code': '00000027', 'sect_name': 'FMCG'},
    # {'sect_code': '00000030', 'sect_name': 'Healthcare'},
    # {'sect_code': '00000032', 'sect_name': 'Infrastructure Developers & Operators'},
    # {'sect_code': '00000067', 'sect_name': 'Insurance'},
    # {'sect_code': '00000034', 'sect_name': 'IT - Software'},
    # {'sect_code': '00000083', 'sect_name': 'Marine Port & Services'},
    # {'sect_code': '00000038', 'sect_name': 'Mining & Mineral products'},
    # {'sect_code': '00000040', 'sect_name': 'Non Ferrous Metals'},
    # {'sect_code': '00000043', 'sect_name': 'Paints/Varnish'},
    # {'sect_code': '00000046', 'sect_name': 'Pharmaceuticals'},
    # {'sect_code': '00000047', 'sect_name': 'Plantation & Plantation Products'},
    # {'sect_code': '00000049', 'sect_name': 'Power Generation & Distribution'},
    # {'sect_code': '00000052', 'sect_name': 'Refineries'},
    # {'sect_code': '00000057', 'sect_name': 'Steel'},
    # {'sect_code': '00000061', 'sect_name': 'Telecomm-Service'},
    # {'sect_code': '00000062', 'sect_name': 'Textiles'},
    # {'sect_code': '00000063', 'sect_name': 'Tobacco Products'},
    # {'sect_code': '00000064', 'sect_name': 'Trading'}
    # ]
    # sect_articles = []
    # for sector in sectors:
    #     sect_code = sector['sect_code']
    #     #sect_name = sector['sect_name']
    #     url=f"http://airrchipapis.cmots.com/api/SectorWiseNews/{sect_code}/20" 
    #     print("Fetching data from URL:", url)
    #     cleaned_articles = get_data_from_url(url)
        # print("Number of articles fetched:", len(cleaned_articles))
        # sect_articles.extend(cleaned_articles)
        # 
    sector_articles=[]
    
    url=f"http://airrchipapis.cmots.com/api/SectorWiseNews/-/50" 
    print("Fetching data from URL:", url)
    sect_articles = []
    cleaned_articles = get_data_from_sec_url(url,bearer_token)
    print("Number of articles fetched:", len(cleaned_articles))
    sect_articles.extend(cleaned_articles) 
    return sect_articles 


def data_sectors():
    sec_arts=sector_articles()
    documents_sec = []
    ids_sec=[]
    existing_ids = set()  # Initialize an empty set
    # for ids in index.list(namespace='news'):
    #     existing_ids.update(ids)
    #print(existing_ids) 
    for article in sec_arts:
        # existing_ids = set()  # Initialize an empty set
        # for ids in index.list(namespace='newsrag'):
        #     existing_ids.update(ids) 
        sno=str(article['sno'])+'s'
        if sno not in existing_ids:

            #date_object = article["date"].split('T')[0]
            date_object = int(article["date"].split('T')[0].replace("-", ""))
    #         text= f"""
    #                 "heading": {article["heading"]},
                    
    #                 "caption": {article["caption"]},
                    
    #                 "article_text": {article["arttext"]}
    #                 """,
            #combined_text = article["heading"] + " " + article["caption"] + " " + article["arttext"]
            #combined_text = f"heading: {article['heading']} caption: [{article['caption']}] text: [{article['arttext']}]"
            combined_text = f'''Sector:{article['sect_name']}: NEWS: {article['heading']} {article['caption']}"'''
            metadatas={
                "date": date_object
            }
            #print(metadatas)
            id_art= article["sno"]
            #print(id_art)
            
            
            documents_sec.append(Document(ids=sno, page_content=combined_text, metadata=metadatas))
            ids_sec.append(sno)
            #docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace=demo_namespace, ids=id_art)
            
            #print("Article inserted into the database:", article["sno"])
        else:
            pass
            #print("Article already exists in the database:", article["sno"])

    #docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace=demo_namespace, ids=ids)
    return documents_sec,ids_sec
    print("Data insertion completed.")


def insert_into_pine(arts):
    docs,ids=data(arts)
    sec_docs,sec_ids=data_sectors()
    #print(docs)
    #print(ids)
    vs.add_documents(documents=docs, ids=ids)
    vs.add_documents(documents=sec_docs, ids=sec_ids)
    #docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name, namespace=demo_namespace, ids=ids)
    #docsearch = Pinecone.from_documents(sec_docs, embeddings, index_name=index_name, namespace=demo_namespace, ids=sec_ids)
    print("inserted",len(ids),len(sec_ids))
    return "Inserted " + str(len(ids)) + str(len(sec_ids))


####################


def get_sentiment_keywords():
    sentiment_keywords = {
        "VERY BULLISH": [],
        "BULLISH": [],
        "NEUTRAL": [],
        "BEARISH": [],
        "VERY BEARISH": []
    }

    Very_Bullish = [
        'record high', '52-week', 'high', 'orders', 'roof', 'breakout', 'surge', 'all-time high', 
        'strong momentum', 'robust earnings', 'positive outlook', 'bullish sentiment', 'significant gains', 
        'exceeds expectations', 'bull market', 'explosive growth', 'stellar performance', 'jump', 'highest', 
        'Breakthrough', 'surges', 'highest-ever', "'strongly", 'Robustly', '\x18optimistic\x19', "'bullish", 
        'Higher-than-expected', 'bullish', 'hyper-growth', 'performance', 'Highest', 'breakthrough', 
        'Surges', 'Highest-Ever', 'strongly', "'earnings", 'positivity', 'Bullish\x19', "'Significant", 
        'higher-than-expected', 'high-growth', 'performance\x19', '\x18highest', 'break-out', 
        "surge", 'All-Time', 'Strongly', "'Earnings", 'Positivity', '\x18Bullish', "'significant", 
        'Higher-Than-Expected', 'Bull\x19', 'High-growth', 'high-performance', 'record-highs', 'breakout-traders', 
        "'Surge", 'all-time', "'Momentum", "'robust", "'Positive", 'significant', 'exceeded', 'Bull', 'Growth', 
        'Stellar', 'record-high', 'breakouts', 'surge\x19', 'All-time', 'high-momentum', "'Robust", "'positive", 
        'Bullish', '\x18significant', 'more-than-expected', 'bull\x19', 'growth\x19', 'stellar', 'Record-High', 
        'Breakout', 'Surge', "'all-time", 'Momentum', 'Robust', 'Positives', 'Bullishness', 'Significant', 
        'Exceeds', 'bull', 'well-performing', 'Record-high', 'breakout', 'surge', 'all-time-high', 
        'momentum', 'positives', 'bullishness', 'significant', 'exceeds', 'Bull', 
        'performance',"wins"
    ]
    
    Bullish = [
        'upward trend', 'bags', 'climbs', 'gain', 'Spurt', 'receives', 'solid growth', 'strong', 'positive', 
        'optimistic outlook', 'favorable conditions', 'rising demand', 'demand','improving fundamentals', 
        'positive economic indicators', 'increased revenue', 'bullish indicators', 'encouraging news', 
        'positive developments', 'up', 'Highest', 'Breakthrough', 'Surges', 'Highest-Ever', "'strongly", 
        'Robustly', 'optimistic\x19', "'bullish", 'gains', 'Higher-Than-Expected', 'Bullish', 'hyper-growth', 
        'performance', 'highest', 'breakthrough', 'surges', 'Highest-ever', 'Strongly', "'earnings", 'positivity', 
        '\x18Bullish', "'significant", 'higher-than-expected', 'High-growth', 'performance\x19', 
        '\x18highest', 'break-out', "'Surge", 'All-Time', 'strongly', "'Earnings", 'Positivity', "'Significant", 
        'Higher-than-expected', 'bull\x19', 'high-growth', 'high-performance', 'record-highs', 'breakout-traders', 
        "'surge", 'all-time', "'Momentum", "'Robust", "'Positive", 'Bullish\x19', '\x18significant', 'exceeded', 
        'Bull\x19', '\x18Growth', 'stellar', 'Record-High', 'breakouts', 'surge\x19', 'All-time', 'high-momentum', 
        "'Robust", "'positive", 'bullish', 'Significant', 'more-than-expected', '\x18Bull\x19', 'growth\x19', 
        'Stellar', 'record-high', 'breakout', 'surge', "'all-time", 'momentum', 'robust', 'positives', 
        'Bullishness', 'significant', 'exceeds', 'bull', 'growth', 'well-performing', 'Record-high', 'Breakout', 
        'Surge', 'all-time-high', 'Momentum', 'Robust', 'Positives', 'bullishness', '\significant', 'Exceeds', 
        'Bull', 'Growth', 'performance','rally','recommends','dividend',"split",'gainer','higher'
    ]
    
    Bearish = [
        'downward trend', 'sell-off', 'bearish sentiment', 'negative outlook', 'underperforming', 
        'weak earnings', 'poor performance', 'falling demand', 'downturn', 'losses', 'bear market', 
        'concerns weigh on', 'slump', 'losers', 'trending', 'exit', 'off', 'despairs', 'downturn', 
        'downgrades', 'underwhelms', 'results', 'slump', 'damage','weigh', 'weak', 'lowest', 'divest', 
        'upside', 'losing', 'bad', 'sell-off', 'economy', 'regains', 'lower', 'Markets', 'problems', 'nill', 
        'dismal', 'down', 'weakens', 'sells', 'affect', 'marginally', 'lost', 'markets', 'stresses', 
        'easing', 'selling', 'Bullish', 'slows', 'poor', 'shortage', 'loss-making', 'ETMarkets', 
        'fears', 'sell', 'bullish', 'downside', 'outperform', 'matters', 'trend', "SELL",
        'loss', 'bearish', 'earnings', 'losses', 'worry','underperforms','drops','resigns','fall','losers'
    ]
    
    Very_Bearish = [
        'plummet', 'crash', 'meltdown', 'Slides', 'steep decline', 'slip', 'worst performance', 'bleak outlook', 
        'catastrophic losses', 'panic selling', 'sharp drop', 'dire situation', 'extreme volatility', 'collapse', 
        'massive sell-off', 'Plunges', 'melts', 'decline', 'WORST', 'Gloom', 'catastrophe', "'panic", 
        'dropped', 'plight', 'Extreme', 'Collapsed', 'Selloffs', 'plummeting', 'crash', 'melted', 'dwindled', 
        'Worst', 'gloom', "'Losses", 'situation', "'Extreme", 'collapsed', 'selloff', 
        'Plummeting', "'Crash", 'melt', 'dwindling', 'worst', 'pessimistic', 'Losses',
        'Situation', "'extreme", 'Collapses', 'Selloff', 'Plummets', "crash", 'Melt', 'Dwindling', 'Worst', 
        'Pessimistic', 'losses', 'panic', 'volatilty', 'collapses', 'sell-offs', 
        'plummets', 'Crash', 'meltdowns', 'worst', 'pessimistic', "'Catastrophic", 'Panic',
        "dire", 'volatility', 'collapsing', 'sell-off', 'Plummet', 'crash', 'Meltdown',
        'Worst-Performing', 'Bleak', "'catastrophic", 'Panic', 'Dire', 'volatility', 'Collapse', 
        'Sell-Off', 'plummet', 'crash', 'meltdown', 'worst-performing', 'bleak', 'catastrophic', 
        'panic-buy', 'dire', 'Volatility', 'Sell-off'
    ]

    Neutral=['table','board','meeting','discuss','schedules','declare','convene','conduct']

    sentiment_keywords["VERY BULLISH"].extend(Very_Bullish)
    sentiment_keywords["BULLISH"].extend(Bullish)
    sentiment_keywords["BEARISH"].extend(Bearish)
    sentiment_keywords["VERY BEARISH"].extend(Very_Bearish)
    sentiment_keywords['NEUTRAL'].extend(Neutral)

    return sentiment_keywords




def classify_sentiment_h(headline):
    sentiment_keywords=get_sentiment_keywords()
    #print(sentiment_keywords)
    lemmatizer = WordNetLemmatizer()

# Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Tokenize the headline and convert to lowercase
    words = word_tokenize(headline.lower())
    #print(words)
    
    # Remove stopwords and lemmatize the remaining words
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    #print(filtered_words)
    
    # Convert sentiment words to lowercase
    expanded_keywords_lower = {category.lower(): [word.lower() for word in keywords] for category, keywords in sentiment_keywords.items()}
    
    # Initialize sentiment counts
    sentiment_counts = {sentiment: 0 for sentiment in expanded_keywords_lower}
    
    # Count the occurrences of each sentiment's keywords
    for sentiment, keywords in expanded_keywords_lower.items():
        for keyword in keywords:
            if keyword in filtered_words or keyword in words:
                sentiment_counts[sentiment] += 1
                #print(f"Keyword matched: {keyword} for sentiment: {sentiment}")
    
    # Determine the sentiment with the highest count
    max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    
    # If no keywords matched, return "Neutral"
    if sentiment_counts[max_sentiment] == 0:
        return 0
    
    return max_sentiment.capitalize()


def get_sentiment(headline):
    #llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")#gpt-3.5-turbo-0613,gpt-3.5-turbo-16k,gpt-4o-2024-05-13
    templete1 = '''You are an experienced stock market investor with expertise in analyzing news headlines for market sentiment.
    Your task is to evaluate the sentiment of the following stock market news headline: {news}.
    Classify the sentiment into one of the following five categories: VERY BULLISH, BULLISH, NEUTRAL, BEARISH, VERY BEARISH. 
    Your analysis should be precise and based solely on the headline provided. The output should be a single sentiment category. Do not include any additional information.
    '''
    news1 = PromptTemplate(template=templete1, input_variables=["news"])
    #GPT4o_mini=ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
    llm_chain_= LLMChain(prompt=news1, llm=GPT4o_mini)
    response=llm_chain_.predict(news=headline)
    return response


def get_sno():
    db_url=psql_url
    conn = psycopg2.connect(db_url)

    # Create a cursor object
    cur = conn.cursor()
    cur.execute('SELECT sno FROM public."newsSentiment"')

    # Fetch all the results
    sno_list = [row[0] for row in cur.fetchall()]

    # Now, 'sno_list' contains all the 'sno' values from the table
    return sno_list


def insert_sentiment(sno, sentiment_value,snos):
    try:
        # Connect to your postgres DB
        db_url=psql_url
        conn = psycopg2.connect(db_url)
        
        # Create a cursor object
        cur = conn.cursor()
        
        # Check if sno is already present
        #cur.execute('SELECT 1 FROM public."newsSentiment" WHERE sno = %s', (sno,))
        #if cur.fetchone() is not None:
        snos_set = set(snos)
        if int(sno) in snos_set:
            print(f"sno {sno} already exists. Skipping insert.")
        else:
            #data.append(sno,sentiment_value)

            cur.execute('INSERT INTO public."newsSentiment" (sno, sentiment) VALUES (%s, %s)', (sno, sentiment_value))
            # cur.execute('''
            #             INSERT INTO public."newsSentiment" (sno, sentiment)
            #             SELECT cmn."capitalMarketNewsId", %s
            #             FROM public."capitalMarketNews" cmn
            #             WHERE cmn."capitalMarketNewsId" = %s
            #         ''', (sentiment_value, sno))
            
            #print(f"Inserted sno {sno} with sentiment '{sentiment_value}'.")

        # Commit the transaction
        conn.commit()
        
        # Close the cursor and connection
        cur.close()
        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")  


#data_snos=[]
def sno_check(sno, sentiment_value,snos):
    data_snos=[]
    snos_set = set(snos)
    if int(sno) in snos_set:
        print(f"sno {sno} already exists. Skipping insert.")
    else:
        d=sno,sentiment_value
        data_snos.append(d)
        #cur.execute('INSERT INTO public."newsSentiment" (sno, sentiment) VALUES (%s, %s)', (sno, sentiment_value))
        # cur.execute('''
        #             INSERT INTO public."newsSentiment" (sno, sentiment)
        #             SELECT cmn."capitalMarketNewsId", %s
        #             FROM public."capitalMarketNews" cmn
        #             WHERE cmn."capitalMarketNewsId" = %s
        #         ''', (sentiment_value, sno))
        
        #print(f"Inserted sno {sno} with sentiment '{sentiment_value}'.")

    # Commit the transaction
    # conn.commit()
    
    # # Close the cursor and connection
    # cur.close()
    # conn.close()
    return data_snos


def insert_data(data):
    # Connect to your database
    db_url=psql_url
    conn = psycopg2.connect(db_url)
    
    # Create a cursor object
    cur = conn.cursor()
    
    # Construct the SQL query
    query = 'INSERT INTO public."newsSentiment" (sno, sentiment) VALUES (%s, %s)'
    
    try:
        # Insert data into the database
        cur.executemany(query, data)
        
        # Commit the transaction
        conn.commit()
        print("Data inserted successfully!")
        
    except Exception as e:
        # Rollback the transaction if an error occurs
        conn.rollback()
        print("Error inserting data:", e)
        
    finally:
        # Close the cursor and connection
        cur.close()
        conn.close()


def insert_sentiment_indb(arts):
    data_snos=[]
    # arts=get_articles()
    snos=get_sno()
    #print(len(snos))
    today_date = datetime.now().strftime("%m/%d/%Y").lstrip("0")
    month, day, year = today_date.split('/')
    # Remove leading zeros from the 
    day = str(int(day))
    # Reconstruct the date string
    today_date = f"{month}/{day}/{year}"
    #print(today_date)
    for article in arts:
        if int(article['sno']) in snos:
            print(article['sno'] + " already existed")
        else:
            date=article['date'].split()[0]
            #print(date)
            if date == today_date or date != today_date:
                sno=article['sno']
                headline = article['heading']
                #print(headline)
                sent=classify_sentiment_h(headline)
                #print(sent)
                if sent == 0:
                    sentiment_llm=get_sentiment(headline)
                    #print(sentiment_llm)
                    sentiment=sentiment_llm.capitalize()
                    #print(sentiment)
                else:
                    sentiment=sent.capitalize()
                    #print(sentiment)
                #sentiment=get_sentiment(headline)
                #insert_sentiment(sno,sentiment,snos)
                #dat=sno_check(sno,sentiment,snos)
                d=sno,sentiment
                data_snos.append(d)
    dat=data_snos
    #print(dat)
    insert_data(dat)
    return len(dat)



@router.post("/create_pine")
def create_pine(ai_key_auth: str = Depends(authenticate_ai_key)):
    arts=get_articles()
    try:
        res=insert_into_pine(arts) 
        res1=insert_sentiment_indb(arts)
        #return JSONResponse(status_code=201, content={"sentiment":res1})
        return JSONResponse(status_code=201, content={"message": "pinecone inserted", "sentiment": res1})
    except Exception as e:
        # If there's an error, return an appropriate error message and status code
        raise HTTPException(status_code=500, detail=f"Error creating resource: {str(e)}")


# create_pine()

# get_sentiment("hi")