import os
import json
import asyncio
import aiohttp
import trafilatura
import requests
import re
from urllib.parse import urlparse
import asyncpg
import pandas as pd
from openai import OpenAI
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
from datetime import datetime

from dotenv import load_dotenv
from config import (
    MAX_WEBPAGE_CONTENT_TOKENS,
    MAX_EMBEDDING_TOKENS,
    MAX_PAGES,
    MAX_SCRAPED_SOURCES,
    encoding
)

load_dotenv()

pg_ip = os.getenv('PG_IP_ADDRESS')
psql_url = os.getenv('DATABASE_URL')

class BraveNewsSearcher:
    """Enhanced Brave Search implementation for news with PostgreSQL integration."""
    
    BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, brave_api_key: str):
        if not brave_api_key:
            raise ValueError("Brave API key is required for BraveNewsSearcher.")
        self.brave_api_key = brave_api_key
        
        # Financial domain list for filtering
        self.domain_list = [
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

    def _url_in_domain_list(self, url):
        """Check if URL is from a trusted financial domain."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return any(domain.endswith(d) for d in self.domain_list)

    async def _fetch_and_parse_url_async(self, url: str) -> str:
        """Fetch content from URL using aiohttp and extract main text using trafilatura."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response.raise_for_status()
                    text_content = await response.text()

            extracted_text = trafilatura.extract(text_content, include_comments=False, include_tables=False)

            if extracted_text:
                # Truncate extracted text to avoid excessively long content
                tokens = encoding.encode(extracted_text)
                if len(tokens) > MAX_WEBPAGE_CONTENT_TOKENS:
                    extracted_text = encoding.decode(tokens[:MAX_WEBPAGE_CONTENT_TOKENS]) + "..."
                print(f"DEBUG: Successfully fetched and parsed URL: {url} ({len(extracted_text)} chars)")
                return extracted_text
            else:
                print(f"WARNING: Could not extract content from URL: {url}")
                return ""
        except Exception as e:
            print(f"WARNING: Failed to fetch/parse URL {url}: {str(e)}")
            return ""

    def _extract_relevant_text(self, brave_results: dict) -> list[dict]:
        """Extract relevant text snippets from Brave Search API results."""
        extracted_data = []

        web_results = brave_results.get('web', {}).get('results', [])
        news_results = brave_results.get('news', {}).get('results', [])

        # Process mixed results if available
        if 'mixed' in brave_results and 'main' in brave_results['mixed']:
            print("DEBUG: Processing 'mixed.main' results from Brave API response.")
            for item_spec in brave_results['mixed']['main']:
                item_type = item_spec.get('type')
                item_index = item_spec.get('index')

                item_to_add = None

                if item_type == 'web' and item_index is not None and item_index < len(web_results):
                    item_to_add = web_results[item_index]
                elif item_type == 'news' and item_index is not None and item_index < len(news_results):
                    item_to_add = news_results[item_index]

                if item_to_add and self._url_in_domain_list(item_to_add.get("url", "")):
                    pub_date = item_to_add.get("page_age")
                    if pub_date:
                        try:
                            datetime.fromisoformat(pub_date)
                        except ValueError:
                            pub_date = None

                    extracted_data.append({
                        "title": item_to_add.get("title", ""),
                        "snippet": item_to_add.get("description", ""),
                        "link": item_to_add.get("url", ""),
                        "publication_date": pub_date
                    })
        else:
            # Fallback to direct processing
            print("DEBUG: Processing direct web and news results.")
            for results, result_type in [(web_results, 'web'), (news_results, 'news')]:
                for item in results:
                    if self._url_in_domain_list(item.get("url", "")):
                        pub_date = item.get("page_age")
                        if pub_date:
                            try:
                                datetime.fromisoformat(pub_date)
                            except ValueError:
                                pub_date = None

                        extracted_data.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("description", ""),
                            "link": item.get("url", ""),
                            "publication_date": pub_date
                        })
        
        return extracted_data

    async def search_and_scrape(self, query_term: str) -> list[dict]:
        """Perform Brave search and scrape content for news articles."""
        all_extracted_content = []
        current_page_num = 1
        total_results_available = float('inf')
        links_encountered = set()

        while current_page_num <= MAX_PAGES:
            if len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                print(f"DEBUG: Reached MAX_SCRAPED_SOURCES ({MAX_SCRAPED_SOURCES}). Stopping.")
                break

            offset = (current_page_num - 1) * 10

            if offset >= total_results_available:
                print(f"DEBUG: Offset {offset} beyond total results {total_results_available}")
                break

            print(f"DEBUG: Fetching Brave API results for query '{query_term}', page {current_page_num}")

            brave_params = {
                "q": query_term,
                "count": 20,
                "country": "in",
                "result_filter": "web,news",
                "freshness": "pm"  # Past month for financial news
            }
            
            brave_headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.BRAVE_API_BASE_URL, 
                        headers=brave_headers, 
                        params=brave_params, 
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as brave_response:
                        brave_response.raise_for_status()
                        brave_results = await brave_response.json()

                if 'query' in brave_results and 'total_results' in brave_results['query']:
                    total_results_available = brave_results['query']['total_results']

                page_extracted_content = self._extract_relevant_text(brave_results)

                if not page_extracted_content:
                    print(f"DEBUG: No more results for query '{query_term}' on page {current_page_num}")
                    break

                for item in page_extracted_content:
                    link = item.get('link')
                    if link and link not in links_encountered and len(all_extracted_content) < MAX_SCRAPED_SOURCES:
                        all_extracted_content.append(item)
                        links_encountered.add(link)
                    elif link in links_encountered:
                        print(f"DEBUG: Skipping duplicate link: {link}")
                    elif len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                        print(f"DEBUG: Reached MAX_SCRAPED_SOURCES limit")
                        break

                if len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                    break

                # Handle pagination
                if brave_results.get('query', {}).get('more_results_available', False):
                    current_page_num += 1
                    await asyncio.sleep(1)  # Rate limiting
                else:
                    print(f"DEBUG: No more results available for query '{query_term}'")
                    break

            except aiohttp.ClientError as e:
                print(f"ERROR: HTTP error fetching Brave API results: {str(e)}")
                break
            except Exception as e:
                print(f"ERROR: Unexpected error: {str(e)}")
                break

        # Scrape content from collected links
        links_to_scrape = [item['link'] for item in all_extracted_content if item.get('link')]
        scrape_tasks = [self._fetch_and_parse_url_async(link) for link in links_to_scrape]
        scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        processed_items = []
        scraped_content_map = {links_to_scrape[i]: scraped_contents[i] for i in range(len(links_to_scrape))}

        for item in all_extracted_content:
            link = item.get('link')
            full_webpage_content = ""

            if link and link in scraped_content_map:
                scraped_result = scraped_content_map[link]
                if not isinstance(scraped_result, Exception):
                    full_webpage_content = scraped_result

            text_to_embed = f"Title: {item['title']}\nSnippet: {item['snippet']}\nFull Content: {full_webpage_content}"
            tokens_to_embed = encoding.encode(text_to_embed)
            if len(tokens_to_embed) > MAX_EMBEDDING_TOKENS:
                text_to_embed = encoding.decode(tokens_to_embed[:MAX_EMBEDDING_TOKENS]) + "..."

            processed_items.append({
                "text_to_embed": text_to_embed,
                "original_item": item,
                "full_webpage_content": full_webpage_content
            })

        return processed_items

    def _process_for_dataframe(self, processed_items: list[dict]) -> pd.DataFrame:
        """Convert processed items to DataFrame format for PostgreSQL storage."""
        news_data = []
        
        for item in processed_items:
            original = item['original_item']
            
            # Extract provider info if available
            heading = "Unknown Source"
            if 'link' in original and original['link']:
                parsed_url = urlparse(original['link'])
                heading = parsed_url.netloc.replace('www.', '').title()
            
            # Format date for PostgreSQL
            source_date = original.get('publication_date')
            if source_date:
                try:
                    # Ensure proper datetime format
                    date_obj = datetime.fromisoformat(source_date.replace('Z', '+00:00'))
                    formatted_date = date_obj.isoformat()
                except:
                    formatted_date = datetime.now().isoformat()
            else:
                formatted_date = datetime.now().isoformat()
            
            news_data.append({
                'source_url': original.get('link', ''),
                'image_url': None,  # Brave API doesn't provide images directly
                'heading': heading,
                'title': original.get('title', '').replace("'", ""),
                'description': original.get('snippet', '').replace("'", ""),
                'source_date': formatted_date,
                'date_published': int(datetime.fromisoformat(formatted_date.replace('Z', '+00:00')).strftime('%Y%m%d'))
            })
        
        if news_data:
            df = pd.DataFrame(news_data)
            # Clean up text fields
            df['title'] = df['title'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', str(x)))
            df['description'] = df['description'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', str(x)))
            df.dropna(subset=['source_url'], inplace=True)
            return df
        
        return pd.DataFrame()


# PostgreSQL integration functions (adapted from original bing_news.py)
async def insert_post(df: pd.DataFrame):
    """Insert news data into PostgreSQL database."""
    db_url = psql_url
    
    try:
        conn = await asyncpg.connect(db_url)
        
        for index, row in df.iterrows():
            source_url = row['source_url']
            image_url = row.get('image_url')
            heading = row['heading']
            title = row['title']
            description = row['description']
            source_date = row['source_date']
            
            # Check if row already exists
            exists = await conn.fetchval("SELECT COUNT(1) FROM source_data WHERE source_url = $1", source_url)
            
            if exists:
                print(f"Row with source_url {source_url} already exists, skipping...")
                continue
            
            # Insert new row
            insert_query = """
                INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            await conn.execute(insert_query, source_url, image_url, heading, title, description, source_date)
        
        print("Data inserted successfully into PostgreSQL")

    except Exception as e:
        print(f"Error inserting into PostgreSQL: {e}")
    finally:
        if conn:
            await conn.close()


def insert_post1(df: pd.DataFrame):
    """Synchronous version of PostgreSQL insert for compatibility."""
    db_url = psql_url

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        for index, row in df.iterrows():
            source_url = row['source_url']
            image_url = row.get('image_url')
            heading = row['heading']
            title = row['title']
            description = row['description']
            source_date = row['source_date']
            
            # Check if row already exists
            cur.execute("SELECT COUNT(1) FROM source_data WHERE source_url = %s", (source_url,))
            exists = cur.fetchone()[0]
            
            if exists:
                continue
            
            # Insert new row
            insert_query = sql.SQL("""
                INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                VALUES (%s, %s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (source_url, image_url, heading, title, description, source_date))
        
        conn.commit()
        print("Data inserted successfully into PostgreSQL (sync)")

    except Exception as e:
        print(f"Error inserting into PostgreSQL (sync): {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# Pinecone integration functions
def initialize_pinecone():
    """Initialize Pinecone client and index."""
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
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


async def data_into_pinecone(df):
    """Store processed data into Pinecone vector database."""
    pc, index_name = initialize_pinecone()
    embeddings = OpenAIEmbeddings()
    documents = []
    ids = []

    for _, row in df.iterrows():
        combined_text = f'''{row["description"]}"'''
        document = Document(
            ids=row["source_url"],
            page_content=combined_text,
            metadata={"url": row["source_url"], "date": row["date_published"]}
        )
        url = row['source_url']
        cleaned_url = re.sub(r'[^a-zA-Z0-9]', '', url).lower()
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


# Main function to get Brave search results
async def get_brave_results(query):
    """Main function to get Brave search results with PostgreSQL integration."""
    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found in environment variables")
        return None, None
    
    searcher = BraveNewsSearcher(brave_api_key)
    
    try:
        processed_items = await searcher.search_and_scrape(query)
        
        if not processed_items:
            print("No results found from Brave search")
            return None, None
        
        # Convert to DataFrame for database storage
        df = searcher._process_for_dataframe(processed_items)
        
        if df.empty:
            print("No valid data for database storage")
            return None, None
        
        # Create articles list for compatibility with existing code
        filtered_articles = []
        for _, row in df.iterrows():
            filtered_articles.append({
                "source_url": row["source_url"],
                "title": row["title"],
                "description": row["description"]
            })
        
        return filtered_articles, df
    
    except Exception as e:
        print(f"Error in get_brave_results: {str(e)}")
        return None, None