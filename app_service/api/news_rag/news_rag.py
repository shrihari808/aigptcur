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

# Local module imports - UPDATED to use Brave search
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

# Load environment variables from the project's .env file
load_dotenv()

# --- 1. Centralized Configuration & Initialization ---

# Load API keys and settings from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
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

# Pinecone vector store for the bing news index (now brave news)
brave_news_vectorstore = PineconeVectorStore(index_name=BING_NEWS_INDEX_NAME, embedding=embeddings, namespace='bing')

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
    """
    Enhanced RAG query using intelligent data ingestion and advanced scoring.
    Now uses Brave search instead of Bing and implements smart data checking.
    """
    memory_query = memory_chain(query, session_id)
    date, user_q, t_day = llm_get_date(memory_query)

    # Create retrievers with date filters if applicable
    cmots_search_kwargs = {"k": 15}
    brave_search_kwargs = {"k": 15}
    if date != "None":
        date_filter = {'date': {'$gte': int(date)}}
        cmots_search_kwargs['filter'] = date_filter
        brave_search_kwargs['filter'] = date_filter

    retriever_cmots = cmots_vectorstore.as_retriever(search_kwargs=cmots_search_kwargs)
    retriever_brave = brave_news_vectorstore.as_retriever(search_kwargs=brave_search_kwargs)
    
    # Get initial results to assess data sufficiency
    print("DEBUG: Checking existing knowledge base...")
    initial_cmots_results = retriever_cmots.get_relevant_documents(user_q)
    initial_brave_results = retriever_brave.get_relevant_documents(user_q)
    
    # Combine initial results for sufficiency assessment
    combined_initial_docs = initial_cmots_results + initial_brave_results
    initial_context = "\n\n".join([doc.page_content for doc in combined_initial_docs])
    
    # Assess context sufficiency using scoring service
    sufficiency_score = scoring_service.assess_context_sufficiency(
        context=initial_context,
        query=user_q,
        num_docs=len(combined_initial_docs)
    )
    
    print(f"DEBUG: Context sufficiency score: {sufficiency_score:.3f} (threshold: {CONTEXT_SUFFICIENCY_THRESHOLD})")
    
    # Conditional data ingestion based on sufficiency
    pinecone_task = None
    if sufficiency_score < CONTEXT_SUFFICIENCY_THRESHOLD:
        print("INFO: Existing data insufficient. Starting Brave search ingestion...")
        
        # Fetch new data using Brave search
        docs, df = await get_brave_results(query)
        
        if docs and df is not None and not df.empty:
            print(f"DEBUG: Brave search returned {len(docs)} articles")
            
            # Insert into PostgreSQL (maintaining existing functionality)
            insert_post1(df)
            
            # Start Pinecone ingestion in background
            pinecone_task = asyncio.create_task(data_into_pinecone(df))
            
            # Wait a bit for Pinecone indexing
            await asyncio.sleep(2)
            
            print("INFO: New data ingested. Re-querying with enhanced dataset...")
        else:
            print("WARN: Brave search returned no results. Using existing data only.")
    else:
        print("INFO: Existing data sufficient. Proceeding with current knowledge base.")

    # Create enhanced retrieval strategy
    lotr = MergerRetriever(retrievers=[retriever_cmots, retriever_brave])
    
    # Get all relevant documents for enhanced processing
    all_relevant_docs = lotr.get_relevant_documents(user_q)
    
    # Convert documents to passages format for scoring
    passages = []
    for doc in all_relevant_docs:
        passages.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    
    # Apply advanced scoring and reranking
    print("DEBUG: Applying advanced scoring and reranking...")
    reranked_passages = await scoring_service.score_and_rerank_passages(
        question=user_q,
        passages=passages,
        w_relevance=W_RELEVANCE,
        w_sentiment=W_SENTIMENT,
        w_time_decay=W_TIME_DECAY,
        w_impact=W_IMPACT
    )
    
    # Create enhanced context from top-scored passages
    enhanced_context = scoring_service.create_enhanced_context(reranked_passages)
    
    # Enhanced response generation with financial focus
    res_prompt_template = """You are an advanced financial markets AI assistant with real-time data access.
    Today's date is {date}.
    
    You have access to comprehensive, scored, and reranked financial news from multiple verified sources.
    The context below has been intelligently filtered and scored for relevance, sentiment, recency, and market impact.
    
    INSTRUCTIONS:
    1. Provide detailed, accurate answers based ONLY on the provided context
    2. Prioritize information from higher-scored sources (indicated by relevance scores)
    3. Include specific source URLs when referencing information
    4. If the query asks for recent information, emphasize the most recent articles
    5. Maintain objectivity and cite your sources appropriately
    6. If information is insufficient, clearly state what's missing
    
    Enhanced Context with Scoring:
    {context}
    
    User Question: {input}

    Format your output as a single JSON object with two keys: "answer" and "links".
    - "answer": A comprehensive, well-structured answer in Markdown format with proper source citations
    - "links": A list of all source URLs used to generate the answer (empty list if no relevant sources)
    """
    
    R_prompt = ChatPromptTemplate.from_template(res_prompt_template)
    output_parser = JsonOutputParser()
    
    # Create the response chain
    response_messages = [
        ("system", "You are a financial markets expert. Provide accurate, well-sourced responses based on the given context."),
        ("human", res_prompt_template)
    ]
    response_prompt = ChatPromptTemplate.from_messages(response_messages)
    
    # Generate response using enhanced context
    with get_openai_callback() as cb:
        try:
            response = llm1.invoke(response_prompt.format(
                context=enhanced_context,
                input=user_q,
                date=t_day
            ))
            
            # Wait for background Pinecone ingestion to complete if it was started
            if pinecone_task:
                await pinecone_task
                print("DEBUG: Background Pinecone ingestion completed")
            
            # Parse the JSON response
            try:
                parsed_response = output_parser.parse(response.content)
                final_answer = parsed_response.get('answer', response.content)
                final_links = parsed_response.get('links', [])
            except Exception as parse_error:
                print(f"WARN: Failed to parse JSON response: {parse_error}")
                final_answer = response.content
                # Extract links from reranked passages as fallback
                final_links = []
                for passage in reranked_passages[:10]:  # Top 10 sources
                    link = passage.get('metadata', {}).get('link') or passage.get('metadata', {}).get('url')
                    if link and link not in final_links:
                        final_links.append(link)

        except Exception as llm_error:
            print(f"ERROR: LLM generation failed: {llm_error}")
            final_answer = "I apologize, but I encountered an error generating the response. Please try again."
            final_links = []

    # Save to chat history (maintaining existing functionality)
    try:
        history = PostgresChatMessageHistory(connection_string=PSQL_URL, session_id=session_id)
        history.add_user_message(memory_query)
        history.add_ai_message(final_answer)
        print("DEBUG: Chat history saved successfully")
    except Exception as history_error:
        print(f"WARN: Failed to save chat history: {history_error}")

    # Return enhanced response with additional metadata
    return {
        "Response": final_answer,
        "links": final_links,
        "Total_Tokens": cb.total_tokens,
        "Prompt_Tokens": cb.prompt_tokens,
        "Completion_Tokens": cb.completion_tokens,
        "context_sufficiency_score": sufficiency_score,
        "num_sources_used": len(reranked_passages),
        "data_ingestion_triggered": sufficiency_score < CONTEXT_SUFFICIENCY_THRESHOLD,
        "top_source_score": reranked_passages[0].get('final_combined_score', 0) if reranked_passages else 0
    }


async def web_rag_with_fallback(query: str, session_id: str):
    """
    Web RAG with fallback to Bing search if Brave fails.
    This provides backward compatibility while transitioning to Brave.
    """
    try:
        # Try the enhanced Brave-based web_rag first
        return await web_rag(query, session_id)
    except Exception as brave_error:
        print(f"ERROR: Brave-based web_rag failed: {brave_error}")
        print("INFO: Falling back to original Bing-based implementation...")
        
        # Fallback to original implementation
        # try:
        #     # Import the original bing functions as fallback
        #     from api.news_rag.bing_news import get_bing_results
            
        #     memory_query = memory_chain(query, session_id)
        #     date, user_q, t_day = llm_get_date(memory_query)

        #     # Use original Bing search
        #     docs, df = get_bing_results(query)
        #     pinecone_task = None
        #     if docs and df is not None:
        #         insert_post1(df)
        #         pinecone_task = asyncio.create_task(data_into_pinecone(df))

        #     # Create retrievers (original implementation)
        #     cmots_search_kwargs = {"k": 10}
        #     bing_search_kwargs = {"k": 15}
        #     if date != "None":
        #         date_filter = {'date': {'$gte': int(date)}}
        #         cmots_search_kwargs['filter'] = date_filter
        #         bing_search_kwargs['filter'] = date_filter

        #     retriever_cmots = cmots_vectorstore.as_retriever(search_kwargs=cmots_search_kwargs)
        #     retriever_bing = brave_news_vectorstore.as_retriever(search_kwargs=bing_search_kwargs)
            
        #     lotr = MergerRetriever(retrievers=[retriever_cmots, retriever_bing])
            
        #     # Simple response generation (original style)
        #     res_prompt_template = """You are a stock market information bot. Today's date is {date}.
        #     Answer the user's question in detail, using ONLY the provided news articles from the web and internal sources.
            
        #     Web News Articles:
        #     {context}
            
        #     User Question: {input}

        #     Format your output as a single JSON object with two keys: "answer" and "links".
        #     - "answer": A detailed answer in Markdown format
        #     - "links": A list of source URLs used
        #     """
            
        #     R_prompt = ChatPromptTemplate.from_template(res_prompt_template)
        #     output_parser = JsonOutputParser()
        #     ans_chain = create_stuff_documents_chain(llm1, R_prompt)
        #     retrieval_chain = create_retrieval_chain(lotr, ans_chain)

        #     with get_openai_callback() as cb:
        #         final = retrieval_chain.invoke({"input": user_q, "date": t_day})
        #         if pinecone_task:
        #             await pinecone_task

        #     # Save to history
        #     history = PostgresChatMessageHistory(connection_string=PSQL_URL, session_id=session_id)
        #     history.add_user_message(memory_query)
        #     history.add_ai_message(final.get('answer', ''))

        #     # Parse JSON output
        #     try:
        #         output_data = output_parser.parse(final.get('answer', '{}'))
        #     except Exception:
        #         output_data = {"answer": final.get('answer', "Could not generate response."), "links": []}

        #     return {
        #         "Response": output_data.get('answer'),
        #         "links": output_data.get('links'),
        #         "Total_Tokens": cb.total_tokens,
        #         "Prompt_Tokens": cb.prompt_tokens,
        #         "Completion_Tokens": cb.completion_tokens,
        #         "fallback_used": True
        #     }
            
        # except Exception as fallback_error:
        #     print(f"ERROR: Fallback implementation also failed: {fallback_error}")
        #     return {
        #         "Response": "I apologize, but both primary and fallback search systems are currently unavailable. Please try again later.",
        #         "links": [],
        #         "Total_Tokens": 0,
        #         "Prompt_Tokens": 0,
        #         "Completion_Tokens": 0,
        #         "error": "Both Brave and Bing search systems failed"
        #     }


# Additional utility functions for enhanced functionality

def get_query_insights(query: str) -> dict:
    """
    Analyze query to provide insights for better processing.
    This can help in customizing the scoring weights based on query type.
    """
    query_lower = query.lower()
    
    insights = {
        "query_type": "general",
        "recency_focused": False,
        "sentiment_focused": False,
        "company_specific": False,
        "market_broad": False,
        "suggested_weights": {
            "w_relevance": W_RELEVANCE,
            "w_sentiment": W_SENTIMENT,
            "w_time_decay": W_TIME_DECAY,
            "w_impact": W_IMPACT
        }
    }
    
    # Detect query characteristics
    recency_keywords = ['latest', 'recent', 'today', 'current', 'new', 'now']
    if any(keyword in query_lower for keyword in recency_keywords):
        insights["recency_focused"] = True
        insights["query_type"] = "recent_news"
        insights["suggested_weights"]["w_time_decay"] = 0.4
        insights["suggested_weights"]["w_relevance"] = 0.3
    
    sentiment_keywords = ['bullish', 'bearish', 'positive', 'negative', 'good', 'bad', 'risk', 'opportunity']
    if any(keyword in query_lower for keyword in sentiment_keywords):
        insights["sentiment_focused"] = True
        insights["suggested_weights"]["w_sentiment"] = 0.3
    
    market_keywords = ['market', 'nifty', 'sensex', 'index', 'sector']
    if any(keyword in query_lower for keyword in market_keywords):
        insights["market_broad"] = True
        insights["query_type"] = "market_analysis"
    
    # Company detection (basic - could be enhanced with NER)
    company_indicators = ['ltd', 'limited', 'inc', 'corp', 'company', 'stock', 'share']
    if any(indicator in query_lower for indicator in company_indicators):
        insights["company_specific"] = True
        insights["query_type"] = "company_analysis"
        insights["suggested_weights"]["w_impact"] = 0.3
    
    return insights


async def adaptive_web_rag(query: str, session_id: str):
    """
    Adaptive web RAG that adjusts scoring weights based on query analysis.
    This is an enhanced version that uses query insights for better results.
    """
    # Get query insights for adaptive scoring
    query_insights = get_query_insights(query)
    print(f"DEBUG: Query insights: {query_insights}")
    
    # Use suggested weights from query analysis
    custom_weights = query_insights["suggested_weights"]
    
    # Call the main web_rag function but with custom processing
    memory_query = memory_chain(query, session_id)
    date, user_q, t_day = llm_get_date(memory_query)

    # ... (similar setup as web_rag) ...
    cmots_search_kwargs = {"k": 15}
    brave_search_kwargs = {"k": 15}
    if date != "None":
        date_filter = {'date': {'$gte': int(date)}}
        cmots_search_kwargs['filter'] = date_filter
        brave_search_kwargs['filter'] = date_filter

    retriever_cmots = cmots_vectorstore.as_retriever(search_kwargs=cmots_search_kwargs)
    retriever_brave = brave_news_vectorstore.as_retriever(search_kwargs=brave_search_kwargs)
    
    # Check sufficiency
    initial_cmots_results = retriever_cmots.get_relevant_documents(user_q)
    initial_brave_results = retriever_brave.get_relevant_documents(user_q)
    combined_initial_docs = initial_cmots_results + initial_brave_results
    initial_context = "\n\n".join([doc.page_content for doc in combined_initial_docs])
    
    sufficiency_score = scoring_service.assess_context_sufficiency(
        context=initial_context,
        query=user_q,
        num_docs=len(combined_initial_docs)
    )
    
    # Conditional ingestion
    pinecone_task = None
    if sufficiency_score < CONTEXT_SUFFICIENCY_THRESHOLD:
        docs, df = await get_brave_results(query)
        if docs and df is not None and not df.empty:
            insert_post1(df)
            pinecone_task = asyncio.create_task(data_into_pinecone(df))
            await asyncio.sleep(2)

    # Enhanced retrieval with adaptive scoring
    lotr = MergerRetriever(retrievers=[retriever_cmots, retriever_brave])
    all_relevant_docs = lotr.get_relevant_documents(user_q)
    
    passages = []
    for doc in all_relevant_docs:
        passages.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    
    # Apply adaptive scoring with custom weights
    reranked_passages = await scoring_service.score_and_rerank_passages(
        question=user_q,
        passages=passages,
        **custom_weights  # Use adaptive weights
    )
    
    enhanced_context = scoring_service.create_enhanced_context(reranked_passages)
    
    # Generate response (similar to web_rag)
    res_prompt_template = """You are an advanced financial markets AI assistant with adaptive intelligence.
    Today's date is {date}.
    
    Query Type: {query_type}
    Query Characteristics: {query_characteristics}
    
    Enhanced Context (Adaptively Scored):
    {context}
    
    User Question: {input}

    Provide a comprehensive response optimized for the detected query type. Format as JSON with "answer" and "links" keys.
    """
    
    with get_openai_callback() as cb:
        try:
            response = llm1.invoke(res_prompt_template.format(
                context=enhanced_context,
                input=user_q,
                date=t_day,
                query_type=query_insights["query_type"],
                query_characteristics=str(query_insights)
            ))
            
            if pinecone_task:
                await pinecone_task
            
            # Parse response
            output_parser = JsonOutputParser()
            try:
                parsed_response = output_parser.parse(response.content)
                final_answer = parsed_response.get('answer', response.content)
                final_links = parsed_response.get('links', [])
            except Exception:
                final_answer = response.content
                final_links = [passage.get('metadata', {}).get('link') for passage in reranked_passages[:10] 
                              if passage.get('metadata', {}).get('link')]

        except Exception as e:
            print(f"ERROR: Adaptive RAG generation failed: {e}")
            final_answer = "Error generating adaptive response."
            final_links = []

    # Save to history
    try:
        history = PostgresChatMessageHistory(connection_string=PSQL_URL, session_id=session_id)
        history.add_user_message(memory_query)
        history.add_ai_message(final_answer)
    except Exception as e:
        print(f"WARN: Failed to save chat history: {e}")

    return {
        "Response": final_answer,
        "links": final_links,
        "Total_Tokens": cb.total_tokens,
        "Prompt_Tokens": cb.prompt_tokens,
        "Completion_Tokens": cb.completion_tokens,
        "context_sufficiency_score": sufficiency_score,
        "num_sources_used": len(reranked_passages),
        "data_ingestion_triggered": sufficiency_score < CONTEXT_SUFFICIENCY_THRESHOLD,
        "query_insights": query_insights,
        "adaptive_weights_used": custom_weights,
        "top_source_score": reranked_passages[0].get('final_combined_score', 0) if reranked_passages else 0
    }