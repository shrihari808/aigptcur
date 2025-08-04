import requests
from datetime import datetime, timedelta
from serpapi import GoogleSearch
import json
import requests
from bs4 import BeautifulSoup
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Summary prompt

summary_template = '''  You are a financial news expert ,summarize the given news articles in 40 words of particular stock {stock} . Write short
 and crisp summary. Answer with a summary.
 {news}

'''
summary_prompt = PromptTemplate(template=summary_template, input_variables=["news","stock"])
# llm_summary = OpenAI(temperature=0.6)
llm1 = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0613", max_tokens=1000)
llm_chain_summary = LLMChain(prompt=summary_prompt, llm=llm1)



headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299"
  }

def extract_metadata(article_links):
        
    article_description = []
    article_keywords = []
    article_title = []
    
    for link in article_links:
        response_ = requests.get(link, headers=headers)
        soup_ = BeautifulSoup(response_.content, "lxml")
        title = soup_.find("title")
        article_title.append(title.text)
        desc_meta_tag = soup_.find('meta', attrs={'name': 'description'})
        if desc_meta_tag:
            description = desc_meta_tag.get('content')
            article_description.append(description)
            
        keywords_meta_tag = soup_.find('meta', attrs={'name': 'keywords'})
        if keywords_meta_tag:
            keywords = keywords_meta_tag.get('content')
            article_keywords.append(keywords)
            
    return article_description,article_title

def get_news(stock_name):
    # Calculate the start and end dates for the last 24 hours

    params = {
        "q": stock_name+"stock news",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "tbm": "nws",
        "location": "Maharashtra, India",
        "num" : "3",
        "sort": "date",
        "tbs": "qdr:d",
        "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    print(results)
    if 'news_results' in results:
        if results['news_results']: 
            links = [result['link'] for result in results['news_results']]

            content = extract_metadata(links)

            summary = llm_chain_summary.run(news=content,stock=stock_name)

            return summary
        else:
            print("No news results available.")
    else:
        print("No news results available.")

    



