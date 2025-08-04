import os
import time
import asyncio
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from typing import Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PostgresChatMessageHistory
from langchain.retrievers import MergerRetriever
from langchain_community.callbacks import get_openai_callback
from langchain_chroma import Chroma

# --- CORRECTED IMPORT: Use ServerlessSpec for modern Pinecone indexes ---
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Local module imports
from api.news_rag.bing_news import get_bing_results, insert_post1, data_into_pinecone
from config import chroma_server_client

# Load environment variables from the project's .env file
load_dotenv()

# --- 1. Centralized Configuration & Initialization ---

# Load API keys and settings from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT') # This is the Cloud Region, e.g., "us-east-1"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PSQL_URL = os.getenv('DATABASE_URL')

# Environment variable validation
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError(
        "FATAL: Missing Pinecone configuration. "
        "Please ensure 'PINECONE_API_KEY' and 'PINECONE_ENVIRONMENT' are set in your .env file."
    )

# Define Pinecone index names and dimension
NEWS_RAG_INDEX_NAME = "newsrag11052024"
BING_NEWS_INDEX_NAME = "bing-news"
PINECONE_INDEX_DIMENSION = 1536

def initialize_pinecone_index(client: PineconeClient, index_name: str, dimension: int, cloud_region: str):
    """Checks if a Pinecone index exists and creates it as a serverless index if it doesn't."""
    if index_name not in client.list_indexes().names():
        print(f"INFO:     Pinecone index '{index_name}' not found. Creating it as a serverless index...")
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            # --- CORRECTED: Use ServerlessSpec to match your account type ---
            spec=ServerlessSpec(
                cloud="aws", # As seen in your screenshot
                region=cloud_region # e.g., "us-east-1"
            )
        )
        print(f"INFO:     Waiting for '{index_name}' to initialize...")
        # Serverless indexes are typically faster to initialize
        time.sleep(10)
        print(f"INFO:     Index '{index_name}' created successfully.")
    else:
        print(f"INFO:     Found existing Pinecone index: '{index_name}'")

# Initialize Pinecone Client and ensure both indexes exist
try:
    print("INFO:     Initializing Pinecone client for News RAG...")
    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
    # Ensure both required indexes are present or created
    initialize_pinecone_index(pinecone_client, NEWS_RAG_INDEX_NAME, PINECONE_INDEX_DIMENSION, PINECONE_ENVIRONMENT)
    initialize_pinecone_index(pinecone_client, BING_NEWS_INDEX_NAME, PINECONE_INDEX_DIMENSION, PINECONE_ENVIRONMENT)
except Exception as e:
    print(f"FATAL:    Failed to initialize or create Pinecone indexes: {e}")
    raise e

# --- Initialize Embeddings, Vector Stores, and LLMs globally ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone vector store for the main news rag index
news_rag_vectorstore = PineconeVectorStore(index_name=NEWS_RAG_INDEX_NAME, embedding=embeddings, namespace='news')

# Pinecone vector store for the bing news index
bing_news_vectorstore = PineconeVectorStore(index_name=BING_NEWS_INDEX_NAME, embedding=embeddings, namespace='bing')

# ChromaDB vector store for CMOTS news
cmots_vectorstore = Chroma(
    client=chroma_server_client,
    collection_name="cmots_news",
    embedding_function=embeddings,
)

# LLMs
llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
llm_date = ChatOpenAI(temperature=0.5, model="gpt-4o-2024-05-13")
llama3 = ChatGroq(temperature=0.2, model="llama3-70b-8192", api_key=GROQ_API_KEY)
llama3_8b = ChatGroq(temperature=0, model="llama3-8b-8192", api_key=GROQ_API_KEY)


# --- 2. Helper & Core Logic Functions ---

def split_input(input_string: str):
    """Splits a string by the first comma."""
    parts = input_string.split(',', 1)
    date = parts[0].strip()
    query = parts[1].strip() if len(parts) > 1 else ""
    return date, query

def llm_get_date(user_query: str):
    """Determines the target date from a user query using an LLM."""
    today = datetime.now().strftime("%Y-%m-%d")
    date_prompt_template = """
        Today's date is {today}.
        User query: "{user_query}"
        From the query, determine the target date.
        - If "today" or "latest" is mentioned, use today's date: {today}.
        - If "recently" is mentioned, use the date 7 days ago.
        - If a specific past date is mentioned, use that date.
        - If no date is mentioned or it's a future date, output "None".
        - If a quarter/year is mentioned, output "None".
        Also, remove the time-related words from the query.
        Format the output ONLY as: YYYYMMDD,modified_user_query
    """
    prompt = ChatPromptTemplate.from_template(date_prompt_template)
    chain = prompt | llm_date
    response = chain.invoke({"today": today, "user_query": user_query}).content
    date, general_query = split_input(response)
    return date, general_query, today

def memory_chain(query: str, session_id: str):
    """Generates a standalone question based on chat history."""
    try:
        connection = psycopg2.connect(PSQL_URL)
        cursor = connection.cursor()
        cursor.execute("SELECT message FROM message_store WHERE session_id = %s ORDER BY id DESC LIMIT 6", (session_id,))
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        chat_history = "\n".join([row[0] for row in reversed(rows)])
    except Exception as e:
        print(f"WARN: Could not fetch chat history for session {session_id}: {e}")
        chat_history = ""

    if not chat_history:
        return query

    contextualize_q_system_prompt = """Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question, including any relevant context from the chat history (like dates or entities). If the question is already standalone, return it as is."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("human", "Chat History:\n{chat_history}\n\nFollow-up Question: {question}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"chat_history": chat_history, "question": query})
    return response.content

def cmots_only(query: str, session_id: str):
    """Performs a RAG query using only the CMOTS news vector store."""
    memory_query = memory_chain(query, session_id)
    date, user_q, t_day = llm_get_date(memory_query)

    search_kwargs = {"k": 20}
    if date != "None":
        search_kwargs['filter'] = {'date': {'$gte': int(date)}}

    retriever = cmots_vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Simplified RAG chain for this specific use case
    qa_system_prompt = """You are a stock news bot. Use the following news articles to answer the question. Provide detailed answers based only on the provided context. Today's date is {date}.
    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm1, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    with get_openai_callback() as cb:
        resp = rag_chain.invoke({"input": user_q, "date": t_day})
    
    return {
        "Response": resp.get("answer", "Could not generate a response."),
        "links": [doc.metadata.get('source') for doc in resp.get('context', []) if doc.metadata.get('source')],
        "Total_Tokens": cb.total_tokens,
        "Prompt_Tokens": cb.prompt_tokens,
        "Completion_Tokens": cb.completion_tokens,
    }

async def web_rag(query: str, session_id: str):
    """Performs a RAG query using both web search (Bing) and CMOTS news."""
    memory_query = memory_chain(query, session_id)
    date, user_q, t_day = llm_get_date(memory_query)

    # Fetch Bing results and start Pinecone ingestion in the background
    docs, df = get_bing_results(query)
    pinecone_task = None
    if docs and df is not None:
        insert_post1(df) # Assumes this is a synchronous DB insert
        pinecone_task = asyncio.create_task(data_into_pinecone(df))

    # Create retrievers with date filters if applicable
    cmots_search_kwargs = {"k": 10}
    bing_search_kwargs = {"k": 15}
    if date != "None":
        date_filter = {'date': {'$gte': int(date)}}
        cmots_search_kwargs['filter'] = date_filter
        bing_search_kwargs['filter'] = date_filter

    retriever_cmots = cmots_vectorstore.as_retriever(search_kwargs=cmots_search_kwargs)
    retriever_bing = bing_news_vectorstore.as_retriever(search_kwargs=bing_search_kwargs)
    
    # Merge results from both sources
    lotr = MergerRetriever(retrievers=[retriever_cmots, retriever_bing])
    
    res_prompt_template = """You are a stock market information bot. Today's date is {date}.
    Answer the user's question in detail, using ONLY the provided news articles from the web and internal sources. Prioritize the most recent articles based on their metadata.
    
    Web News Articles:
    {context}
    
    User Question: {input}

    Format your output as a single JSON object with two keys: "answer" and "links".
    - "answer": A detailed, precise answer in Markdown format. Do not include URLs here.
    - "links": A list of all source URLs used to generate the answer. Return an empty list if no relevant links were found.
    """
    R_prompt = ChatPromptTemplate.from_template(res_prompt_template)
    output_parser = JsonOutputParser()
    ans_chain = create_stuff_documents_chain(llm1, R_prompt)
    retrieval_chain = create_retrieval_chain(lotr, ans_chain)

    with get_openai_callback() as cb:
        final = retrieval_chain.invoke({"input": user_q, "date": t_day})
        # Wait for background ingestion to finish if it was started
        if pinecone_task:
            await pinecone_task

    # Save to history
    history = PostgresChatMessageHistory(connection_string=PSQL_URL, session_id=session_id)
    history.add_user_message(memory_query)
    history.add_ai_message(final.get('answer', ''))

    # Parse the JSON output from the LLM
    try:
        output_data = output_parser.parse(final.get('answer', '{}'))
    except Exception:
        output_data = {"answer": final.get('answer', "Could not generate a valid response."), "links": []}

    return {
        "Response": output_data.get('answer'),
        "links": output_data.get('links'),
        "Total_Tokens": cb.total_tokens,
        "Prompt_Tokens": cb.prompt_tokens,
        "Completion_Tokens": cb.completion_tokens,
    }
