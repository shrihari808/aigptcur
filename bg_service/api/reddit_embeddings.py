import praw
from chromadb.utils import embedding_functions
import datetime
import os
from fastapi import APIRouter, HTTPException
from config import chroma_server_client  # Ensure this import is correct
from dotenv import load_dotenv
import concurrent.futures

load_dotenv(override=True)

router = APIRouter()
subs = [
    "IndiaInvestments",  # General discussions and news (200k+ members)
    "StockMarketIndia",  # Stock market discussions (180k+ members)
    "IndianStockMarket",
    "ThetaGang",  # Options trading strategies (40k+ members)
    "IndianStreetBets",  # High-risk options trading (60k+ members, use with caution)
    "MutualFundsIndia",  # Mutual funds discussions (22k+ members)
    "dividends",
    "unitedstatesofindia",
    "IndiaFinance",  # Personal finance (180k+ members)
    "IndiaSpeaks",
    "FIREIndia",  # Financial Independence, Retire Early (FIRE) (13k+ members)
    "FatFIREIndia",
    "personalfinanceindia",  # Value investing discussions (2.4k+ members)
    "IndiaTax",  # Taxes in India (22k+ members, relevant for financial planning)
]

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

def initialize():
    print("Initializing Reddit and ChromaDB clients...")
    
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: CLIENT_ID or CLIENT_SECRET not set in environment variables.")
        return None, None, None

    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent='airrchip_agent'
        )
    except Exception as e:
        print(f"Failed to initialize Reddit client: {e}")
        return None, None, None

    try:
        if chroma_server_client is None:
            print("Error: chroma_server_client is None. Ensure it is properly configured.")
            return None, None, None

        print(f"client {chroma_server_client}")  # Debug print
        client = chroma_server_client
        em = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-base")
        print(f"em {em}")  # Debug print
        
        # Add error handling for get_or_create_collection
        try:
            post_collection = client.get_or_create_collection(name="post_data", embedding_function=em)
            print("count",post_collection.count())
            comment_collection = client.get_or_create_collection(name="comments", embedding_function=em)
            print("comment count",post_collection.count())
        except Exception as e:
            print(f"Failed to initialize ChromaDB collections: {e}")
            return None, None, None

    except Exception as e:
        print(f"Failed to initialize ChromaDB client: {e}")
        return None, None, None

    print("Initialization complete.")
    return reddit, post_collection, comment_collection


def check_exists(collection, id):
    try:
        result = collection.get(ids=[id])
        return len(result['documents']) > 0
    except:
        return False

def insert_into_database(submission, post_collection, batch_posts):
    if not check_exists(post_collection, submission.id):
        submission_id = submission.id
        title = submission.title
        url = submission.url
        author = str(submission.author)
        created_utc = str(datetime.datetime.utcfromtimestamp(submission.created_utc))
        subreddit_name = str(submission.subreddit)
        content = submission.selftext

        batch_posts.append({
            "document": f'"Title": {title}, "Content": {content}',
            "metadata": {
                "UTC": created_utc,
                "URL": url,
                "Subreddit": subreddit_name,
                "Author": author
            },
            "id": submission_id
        })

def insert_comment(submission, comment, comment_collection, batch_comments):
    if not check_exists(comment_collection, comment.id):
        comment_body = comment.body
        comment_id = comment.id
        comment_utc = str(datetime.datetime.utcfromtimestamp(comment.created_utc))
        
        batch_comments.append({
            "document": comment_body,
            "metadata": {
                "UTC": comment_utc,
                "post_id": submission.id
            },
            "id": comment_id
        })

def process_subreddit(subreddit_name, reddit, post_collection, comment_collection):
    print(f"Processing subreddit: {subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)
    new_submissions = subreddit.new(limit=100)
    batch_posts = []
    batch_comments = []

    try:
        for submission in new_submissions:
            insert_into_database(submission, post_collection, batch_posts)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list():
                insert_comment(submission, comment, comment_collection, batch_comments)
        
        if batch_posts:
            post_collection.add(
                documents=[post["document"] for post in batch_posts],
                metadatas=[post["metadata"] for post in batch_posts],
                ids=[post["id"] for post in batch_posts]
            )
            print(f"Added {len(batch_posts)} new posts to post_collection.")
        
        if batch_comments:
            comment_collection.add(
                documents=[comment["document"] for comment in batch_comments],
                metadatas=[comment["metadata"] for comment in batch_comments],
                ids=[comment["id"] for comment in batch_comments]
            )
            print(f"Added {len(batch_comments)} new comments to comment_collection.")

        print(f"Data written to DB for subreddit: {subreddit_name}")

    except Exception as e:
        print(f"An error occurred in {subreddit_name}: {e}")

def insert_reddit_data():
    reddit, post_collection, comment_collection = initialize()

    if reddit is None or post_collection is None or comment_collection is None:
        print("Initialization failed. Exiting...")
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_subreddit, sub, reddit, post_collection, comment_collection) for sub in subs]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during processing: {e}")

@router.post("/insert-reddit-data")
async def insert_data():
    try:
        print("Starting Reddit data insertion...")
        insert_reddit_data()
        print("Reddit data insertion completed successfully.")
        return {"message": "Reddit data insertion completed successfully."}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))