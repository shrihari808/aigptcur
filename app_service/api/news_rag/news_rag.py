import os
import time
import asyncio
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_postgres import PostgresChatMessageHistory
from langchain.retrievers import MergerRetriever
from langchain_community.callbacks import get_openai_callback
from langchain_chroma import Chroma

from pinecone import Pinecone as PineconeClient, ServerlessSpec

from api.news_rag.brave_news import get_brave_results, insert_post1, data_into_pinecone
from api.news_rag.scoring_service import scoring_service
from config import (
    chroma_server_client,
    CONTEXT_SUFFICIENCY_THRESHOLD,
    W_RELEVANCE,
    W_SENTIMENT,
    W_TIME_DECAY,
    W_IMPACT
)

load_dotenv()

async def cmots_only():
    pass

# --- 1. Centralized Configuration & Initialization ---

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PSQL_URL = os.getenv('DATABASE_URL')

if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("FATAL: Missing Pinecone configuration.")

NEWS_RAG_INDEX_NAME = "newsrag11052024"
BING_NEWS_INDEX_NAME = "bing-news"
PINECONE_INDEX_DIMENSION = 1536

def initialize_pinecone_index(client: PineconeClient, index_name: str, dimension: int, cloud_region: str):
    if index_name not in client.list_indexes().names():
        client.create_index(
            name=index_name, dimension=dimension, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=cloud_region)
        )
        time.sleep(10)

try:
    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
    initialize_pinecone_index(pinecone_client, NEWS_RAG_INDEX_NAME, PINECONE_INDEX_DIMENSION, PINECONE_ENVIRONMENT)
    initialize_pinecone_index(pinecone_client, BING_NEWS_INDEX_NAME, PINECONE_INDEX_DIMENSION, PINECONE_ENVIRONMENT)
except Exception as e:
    raise e

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
news_rag_vectorstore = PineconeVectorStore(index_name=NEWS_RAG_INDEX_NAME, embedding=embeddings, namespace='news')
brave_news_vectorstore = PineconeVectorStore(index_name=BING_NEWS_INDEX_NAME, embedding=embeddings, namespace='bing')
cmots_vectorstore = Chroma(
    client=chroma_server_client,
    collection_name="cmots_news",
    embedding_function=embeddings,
)

llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
llm_date = ChatOpenAI(temperature=0.5, model="gpt-4o-2024-05-13")

# --- 2. Helper & Core Logic Functions ---

def get_query_insights(query: str) -> dict:
    query_lower = query.lower()
    insights = {
        "recency_focused": any(keyword in query_lower for keyword in ['latest', 'recent', 'today', 'current', 'new', 'now'])
    }
    return insights

def split_input(input_string: str):
    parts = input_string.split(',', 1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""

def llm_get_date(user_query: str):
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = ChatPromptTemplate.from_template(
        "Today's date is {today}. User query: \"{user_query}\". "
        "Determine the target date. If 'today' or 'latest', use today. "
        "If 'recently', use 7 days ago. For specific past dates, use that date. "
        "If no date or future date, output 'None'. "
        "Also, remove time-related words from the query. "
        "Format output ONLY as: YYYYMMDD,modified_user_query"
    )
    chain = prompt | llm_date
    response = chain.invoke({"today": today, "user_query": user_query}).content
    return split_input(response) + (today,)

def memory_chain(query: str, session_id: str):
    try:
        connection = psycopg2.connect(PSQL_URL)
        cursor = connection.cursor()
        cursor.execute("SELECT message FROM message_store WHERE session_id = %s ORDER BY id DESC LIMIT 6", (session_id,))
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        chat_history = "\n".join([row[0] for row in reversed(rows)])
    except Exception as e:
        chat_history = ""
    if not chat_history:
        return query
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and a follow-up question, rephrase it to be a standalone question."),
        ("human", "Chat History:\n{chat_history}\n\nFollow-up Question: {question}")
    ])
    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return chain.invoke({"chat_history": chat_history, "question": query}).content

def is_response_insufficient(response_text: str) -> bool:
    """
    Checks if the LLM response is too short or indicates a lack of information.
    """
    if len(response_text) < 150: # Reduced threshold slightly
        return True
    
    insufficient_phrases = [
        "cannot find the information", "no recent financial news",
        "context provided does not contain", "unable to provide",
        "no information available", "based on the context provided, there is no"
    ]
    
    lower_response = response_text.lower()
    for phrase in insufficient_phrases:
        if phrase in lower_response:
            return True
            
    return False

async def web_rag(query: str, session_id: str):
    """
    Performs a two-pass RAG query. 
    Pass 1 uses existing data. 
    Pass 2 triggers a web search if the first pass response is insufficient.
    """
    memory_query = memory_chain(query, session_id)
    date, user_q, t_day = llm_get_date(memory_query)

    search_kwargs = {"k": 15}
    if date != "None":
        search_kwargs['filter'] = {'date': {'$gte': int(date)}}

    retriever_cmots = cmots_vectorstore.as_retriever(search_kwargs=search_kwargs)
    retriever_brave = brave_news_vectorstore.as_retriever(search_kwargs=search_kwargs)
    lotr = MergerRetriever(retrievers=[retriever_cmots, retriever_brave])

    # --- Pass 1: Initial attempt with existing data ---
    print("INFO: Pass 1 - Attempting to answer using existing knowledge.")
    initial_docs = lotr.get_relevant_documents(user_q)
    
    passages = [{"text": doc.page_content, "metadata": doc.metadata} for doc in initial_docs]
    reranked_passages = await scoring_service.score_and_rerank_passages(
        question=user_q, passages=passages
    )
    enhanced_context = scoring_service.create_enhanced_context(reranked_passages)

    res_prompt_template = """You are an advanced financial markets AI assistant. Today's date is {date}.
    Based ONLY on the context provided below, answer the user's question.
    If the context is empty or insufficient, state that you cannot find the information in the provided articles.
    Context:
    {context}
    User Question: {input}
    Format your output as a single JSON object with "answer" and "links" keys.
    """
    R_prompt = ChatPromptTemplate.from_template(res_prompt_template)
    output_parser = JsonOutputParser()
    response_chain = R_prompt | llm1 | output_parser

    with get_openai_callback() as cb:
        try:
            first_pass_response = response_chain.invoke({"context": enhanced_context, "input": user_q, "date": t_day})
            final_answer = first_pass_response.get('answer', "")
            final_links = first_pass_response.get('links', [])
        except Exception:
            final_answer = ""
            final_links = []
    
    total_tokens = cb.total_tokens
    prompt_tokens = cb.prompt_tokens
    completion_tokens = cb.completion_tokens
    data_ingestion_triggered = False

    # --- Self-Correction Quality Check ---
    if is_response_insufficient(final_answer):
        print("INFO: First-pass response is insufficient. Triggering corrective web search (Pass 2).")
        data_ingestion_triggered = True
        
        articles, df = await get_brave_results(query)
        
        if articles and df is not None and not df.empty:
            print("INFO: Web search found new articles. Re-running RAG chain.")
            insert_post1(df)
            
            # **FIX: Run Pinecone ingestion as a background task without blocking.**
            # The 'await' is removed from the sleep call, and we let the task run.
            pinecone_task = asyncio.create_task(data_into_pinecone(df))

            # Allow a very brief moment for new data to be retrievable, but don't block.
            await asyncio.sleep(0.1) 

            all_relevant_docs = lotr.get_relevant_documents(user_q)
            passages = [{"text": doc.page_content, "metadata": doc.metadata} for doc in all_relevant_docs]
            reranked_passages = await scoring_service.score_and_rerank_passages(
                question=user_q, passages=passages
            )
            enhanced_context = scoring_service.create_enhanced_context(reranked_passages)

            with get_openai_callback() as cb2:
                try:
                    second_pass_response = response_chain.invoke({"context": enhanced_context, "input": user_q, "date": t_day})
                    final_answer = second_pass_response.get('answer', "Failed to generate a refined answer.")
                    final_links = second_pass_response.get('links', [])
                except Exception as e:
                    final_answer = "I found new information but encountered an error while processing it."
                    final_links = [p.get('metadata', {}).get('link') for p in reranked_passages[:5] if p.get('metadata', {}).get('link')]

                total_tokens += cb2.total_tokens
                prompt_tokens += cb2.prompt_tokens
                completion_tokens += cb2.completion_tokens
            
            # Ensure the background task is awaited before the function exits if needed,
            # but for this API, we can let it complete in the background.
            # await pinecone_task 
        else:
            print("WARN: Corrective web search found no new articles. Using initial response.")
        
        if is_response_insufficient(final_answer):
            print("WARN: Second pass also resulted in an insufficient answer. Returning a user-friendly message.")
            final_answer = f"I searched for recent information about '{query}' but could not find any specific articles to provide a detailed answer. Please try rephrasing your query or asking about a different topic."
            final_links = []

    # --- Final Response and History Logging ---
    try:
        history = PostgresChatMessageHistory(connection_string=PSQL_URL, session_id=session_id)
        history.add_user_message(memory_query)
        history.add_ai_message(final_answer)
    except Exception as history_error:
        print(f"WARN: Failed to save chat history: {history_error}")

    return {
        "Response": final_answer, "links": final_links, "Total_Tokens": total_tokens,
        "Prompt_Tokens": prompt_tokens, "Completion_Tokens": completion_tokens,
        "context_sufficiency_score": 0,
        "num_sources_used": len(reranked_passages),
        "data_ingestion_triggered": data_ingestion_triggered,
        "top_source_score": reranked_passages[0].get('final_combined_score', 0) if reranked_passages else 0
    }

# The adaptive_web_rag and web_rag_with_fallback functions can be copied from the previous version.
async def web_rag_with_fallback(query: str, session_id: str):
    try:
        return await web_rag(query, session_id)
    except Exception as brave_error:
        print(f"ERROR: Brave-based web_rag failed: {brave_error}")
        return { "Response": "I apologize, but the search system is currently unavailable.", "links": [], "error": str(brave_error) }

async def adaptive_web_rag(query: str, session_id: str):
    # This function can be implemented similarly to the main web_rag but using custom weights from get_query_insights
    return await web_rag(query, session_id) # Fallback to standard for now
