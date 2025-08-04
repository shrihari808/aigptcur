import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import json
import tiktoken
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, APIRouter
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from pinecone import PodSpec, Pinecone as PineconeClient
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import re
import psycopg2
from langchain.retrievers import (
    MergerRetriever,
)
from langchain.docstore.document import Document
from datetime import datetime
from langchain_pinecone import Pinecone
import google.generativeai as genai
import requests
from datetime import datetime
import re
import os
#from create_pine import insert_into_pine
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import (
    PostgresChatMessageHistory,
)
from langchain.memory import ConversationBufferWindowMemory

#from langchain_postgres import PostgresChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from psycopg2 import sql
from contextlib import contextmanager
from config import chroma_server_client,llm_date,llm_stream,vs,GPT4o_mini
from langchain_chroma import Chroma
import time
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import JSONResponse




from streaming.bing_stream import get_bing_results,insert_post1
from streaming.reddit_stream import fetch_search_red,process_search_red,insert_red
from streaming.yt_stream import get_data,get_yt_data_async
# from fund import agent2




from dotenv import load_dotenv
load_dotenv(override=True)

openai_api_key=os.getenv('OPENAI_API_KEY')
pg_ip=os.getenv('PG_IP_ADDRESS')
pine_api=os.getenv('PINECONE_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')
psql_url=os.getenv('DATABASE_URL')
node_key=os.getenv('node_key')

# from langchain.globals import set_debug

# set_debug(True)



# index_name = "news"
# demo_namespace='newsrag'
# index_name = "newsrag11052024"
# demo_namespace='news'
# embeddings = OpenAIEmbeddings()

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)



# pc = PineconeClient(
#  api_key=pine_api
# )


# index = pc.Index(index_name)

# #demo_namespace='newsrag'
# docsearch1 = Pinecone(
#     index, embeddings, "text", namespace=demo_namespace
# )


#llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",stream_usage=True,streaming=True)
#gpt-3.5-turbo-1106,gpt-3.5-turbo-16k
#llm1=ChatOpenAI(temperature=0.5, model="gpt-4o-2024-05-13")
# #gpt-3.5-turbo-instruct,gpt-4-turbo,gpt-4o-2024-05-13
#llm1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro")




async def llm_get_date(user_query):
    today = datetime.now().strftime("%Y-%m-%d")
    #print(today)
    date_prompt = """
        Today's date is {today}.
        Here's the user query: {user_query}
        Using the above given date for context, figure out which date the user wants data for.
        If the user query mentions "today" ,then use the above given today's date. Output the date in the format YYYYMMDD.
        If the user query mentions "yesterday"or "trending",output 1 day back date from todays date in YYYYMMDD.
        If the user is asking about recently/latest news in user query .output 7 days back date from todays date in YYYYMMDD.
        If the user is aksing about specifc time period in user query from past. output the start date the user mentioned in YYYYMMDD format.
        If the user doesnot mention any date in user query or asking about upcoming date outpute date as  "None" and If the user mention anything about quater and year output date as "None".

        Also, remove time-related references from the user query, ensuring it remains grammatically correct.

        Format your output as:
        YYYYMMDD,modified_user_query

        """
    D_prompt = PromptTemplate(template=date_prompt, input_variables=["today","user_query"])
    llm_chain_date= LLMChain(prompt=D_prompt, llm=llm_date)
    response=await llm_chain_date.arun(today=today,user_query=user_query)
    response = response.strip()
    #print(response)
    date, general_user_query = split_input(response)

    return date,general_user_query,today


def split_input(input_string):
    # Split the input string at the first comma
    parts = input_string.split(',', 1)
    # Assign the parts to date and general_user_query
    date = parts[0].strip()
    general_user_query = parts[1].strip() if len(parts) > 1 else ""
    return date, general_user_query


def count_tokens(text, model_name="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

async def memory_chain(query,m_chat):
    # s_id=str(session_id)
    # connection = psycopg2.connect(psql_url)

    # # Create a cursor
    # cursor = connection.cursor()

    # # View the table
    # table_name = 'message_store'
    # cursor.execute("SELECT message FROM message_store WHERE session_id = %s", (s_id,))
    # rows = cursor.fetchall()

    # # for row in rows:
    # #     print(row)

    # # Close the cursor and connection
    # cursor.close()
    # connection.close()


    # #chat=rows[:4]
    # chat=[row[0]['data']['content'] for row in rows[-4:]]

    #print(type(chat))
    # print(chat[0][0]['data']['content'])
    # return chat
    contextualize_q_system_prompt = """Given a chat history and the user question \
    which might reference context in the chat history, formulate a standalone question if needed include time/date part also based on user previous question.\
    which can be understood without the chat history. Do NOT answer the question,
    If user question contains only stock name or stock ticker reformulate question as recent news of that stock.
    If user question contains current news / recent trends reformulate question as todays market news or trends.
    just reformulate it if needed and if the user question doesnt have any relevancy to chat history return as it is. 
    chat history:{chat_his}
    user question:{query}
    
    
    """

    c_q_prompt=PromptTemplate(template=contextualize_q_system_prompt,input_variables=['chat_his','query'])
    #llm=ChatOpenAI(model="gpt-4o-2024-05-13",temperature=0.5)
    memory_chain=LLMChain(prompt=c_q_prompt,llm=llm_date)
    res=await memory_chain.arun(query=query,chat_his=m_chat)
    #print(res)

    return res

async def query_validate(query,session_id):
    res_prompt = """
    You are a highly skilled indian stock market investor and financial advisor. Your task is to validate whether a given question is related to the stock market or finance or elections or economics or general private listed companies. Additionally, if the new question is a follow-up then only use chat history to determine its validity.
    If question is asking about latest news about any company or current news or just company or trending news of any company consider it as valid question.
    Given question : {q}
    chat history : {list_qs}
    Output the result in JSON format:
    "valid": Return 1 if the question is valid, otherwise return 0.
    """


    R_prompt = PromptTemplate(template=res_prompt, input_variables=["list_qs","q"])
    # llm_chain_res= LLMChain(prompt=R_prompt, llm=GPT4o_mini)
    chain = R_prompt | GPT4o_mini | JsonOutputParser()


    db_url=psql_url
    conn = psycopg2.connect(db_url)

    # Create a cursor object
    cur = conn.cursor()

    # Execute the SQL query with the session_id as a string
    s_id=str(session_id)
    cur.execute("SELECT message FROM message_store WHERE session_id = %s", (s_id,))
    messages = cur.fetchall()
    chat=[row[0]['data']['content'] for row in messages[-2:]]
    m_chat=[row[0]['data']['content'] for row in messages[-4:]]
    h_chat=[row[0]['data']['content'] for row in messages[-6:]]
    cur.close()
    conn.close()
    #print(query)
    with get_openai_callback() as cb:
        input_data = {
            "list_qs": chat,
            "q":query
        }
        #res=llm_chain_res.predict(query=q)
        res=await chain.ainvoke(input_data)
    return res['valid'],cb.total_tokens,m_chat,h_chat



# def set_ret():
#     embeddings = OpenAIEmbeddings()
#     index_name = "newsrag11052024"
#     demo_namespace='news'

#     index = pc.Index(index_name)

#     #demo_namespace='newsrag'
#     docsearch_cmots = Pinecone(
#         index, embeddings, "text", namespace=demo_namespace
#     )

#     index_name1 = "bing-news"
#     demo_namespace1='bing'

#     index1 = pc.Index(index_name1)
#     embeddings = OpenAIEmbeddings()

#     #demo_namespace='newsrag'
#     docsearch_bing = Pinecone(
#         index1, embeddings, "text",namespace=demo_namespace1
#     )

#     return docsearch_bing,docsearch_cmots

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


def store_into_db_no(pid,ph_id,result_json):
    #db_url = f"postgresql://postgresql:1234@{pg_ip}/frruitmicro"
    db_url=psql_url
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Update all existing entries to set isactive to false
 
            result_json_str = json.dumps(result_json)
            #print(result_json_str)

            cur.execute("""
                INSERT INTO "streamingData" (prompt_id, prompt_history_id, source_data)
                VALUES (%s, %s, %s)
            """, (pid, ph_id, result_json_str))

            # Commit the transaction
            conn.commit()
            #print(f"Data inserted successfully with prompt_id: {pid}")

async def store_into_db(pid, ph_id, result_json):
    # Database connection URL
    db_url = psql_url

    # Convert result_json to a JSON string
    result_json_str = json.dumps(result_json)

    # Establish an async connection to the database
    conn = await asyncpg.connect(db_url)
    try:
        # Insert into the streamingData table
        await conn.execute("""
            INSERT INTO "streamingData" (prompt_id, prompt_history_id, source_data)
            VALUES ($1, $2, $3)
        """, pid, ph_id, result_json_str)
    finally:
        # Close the connection
        await conn.close()

def get_user_credits(user_id):
    url = f"https://api.frruit.co/api/users/getUserCredits?user_id={user_id}"
    headers = {
        "x-api-key": node_key
    }

    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Convert response to JSON
        return data['data']['remainingCredits']
    else:
        print(f"Error: {response.status_code}")


def store_into_userplan(user_id, count):
    #db_url = f"postgresql://postgresql:1234@{pg_ip}/frruitmicro"
    db_url=psql_url
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Check if user_id exists and is active
            cur.execute("""
                SELECT credits_used FROM "user_plans"
                WHERE user_id = %s AND \"isActive\" = true
            """, (user_id,))
            result = cur.fetchone()

            if result:
                # If user exists and is active, update credits_used
                current_credits = result[0]
                new_credits = current_credits + count
                cur.execute("""
                    UPDATE "user_plans"
                    SET credits_used = %s, "updatedAt" = %s
                    WHERE user_id = %s AND \"isActive\" = true
                """, (new_credits, current_time, user_id))
                #print(f"Credits updated successfully for user_id: {user_id}")
            else:
                # Insert new record if user_id does not exist or is inactive
                cur.execute("""
                    INSERT INTO "user_plans" (user_id, credits_used, "is_active")
                    VALUES (%s, %s, true)
                """, (user_id, count))
                #print(f"Data inserted successfully for user_id: {user_id}")

            # Commit the transaction
            conn.commit()


def insert_credit_usage_no(user_id, plan_id, credit_used):
    db_url=psql_url
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Get the current time in the desired format
            current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'

            # Insert into the credit_usage table
            cur.execute("""
                INSERT INTO credit_usage (user_id, plan_id, credit_used, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, plan_id, credit_used, current_time, current_time))

            # Commit the transaction
            conn.commit()
            #print(f"Data inserted successfully for user_id: {user_id}, plan_id: {plan_id}")

import asyncpg
async def insert_credit_usage(user_id, plan_id, credit_used):
    # Database connection URL
    db_url = psql_url

    # Get the current time in the desired format
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'

    # Create a connection pool for efficient resource management
    conn = await asyncpg.connect(db_url)
    try:
        # Insert into the credit_usage table
        await conn.execute("""
            INSERT INTO credit_usage (user_id, plan_id, credit_used, "createdAt", "updatedAt")
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, plan_id, credit_used, current_time, current_time)
    finally:
        # Close the connection
        await conn.close()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    history = PostgresChatMessageHistory(
    connection_string = psql_url,
    session_id=session_id,
    )
#memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2, chat_memory=history)
    return history.messages

class InRequest(BaseModel):
    query: str


app = FastAPI()
cmots_rag = APIRouter()
web_rag = APIRouter()
red_rag=APIRouter()
yt_rag=APIRouter()


# client=chroma_server_client
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vs= Chroma(
#     client=client,
#     collection_name="cmots_news",
#     embedding_function=embeddings,)

AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )



@cmots_rag.post("/cmots_rag")
async def cmots_only(
    request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...),
    ai_key_auth: str = Depends(authenticate_ai_key)
):
    query = request.query
    # Start the timer
    # Call the query_validate function
    valid, v_tokens,m_chat,h_chat=await query_validate(query, session_id)
    #print(valid)

    # valid,v_tokens,m_chat=query_validate(query,session_id)
    #print(valid)
    if valid == 0:
        #pass
        docs, his, memory_query, date = '', '', '', ''
        #i want to stream "not valid question"
    else:
        memory_query=await memory_chain(query,m_chat)
        print(memory_query)
        date,user_q,t_day=await llm_get_date(memory_query)
        #print(date)
        user_q=memory_query
        # text_field = "text"  
        # vectorstore = PineconeVectorStore(  
        #     index, embeddings, text_field  ,namespace=demo_namespace
        # )  
        # if date == "None":
        #     retriever = vs.as_retriever(search_kwargs={"k": 10}
        #                                 )
        # else:
        #     retriever = vs.as_retriever(search_kwargs={"k": 10, 
        #                                                 'filter': {'date':{'$gte': int(date)}}
        #                                                }
        # 
        #                                 )
        #his =get_session_history(str(session_id))
        his =h_chat
        try:
            if date == 'None':
                results = vs.similarity_search_with_score(
                    memory_query,
                    k=10,

                )
                docs=[doc[0].page_content for doc in results]
            #docs = retriever.invoke(memory_query)
            else:
                results = vs.similarity_search_with_score(
                    memory_query,
                    k=10,
                    filter={"date": {"$gte": int(date)}},

                )
                #print(results)
                docs=[doc[0].page_content for doc in results]
        except Exception as e:
            docs = None  # or [] if you prefer an empty list
            # Optionally, log the exception or print it for debugging
            print(f"An error occurred: {e}")




        res_prompt = """
        cmots news articles :{cmots} 
        chat history : {history}
        Today date:{date}
        You are a stock news and stock market information bot. 
        
        use the date provided in the metadata to answer the user query if the user is asking in specific time periods.
        If the same question {input} present in chat_history ignore that answer present in chat history dont consider that answer while generating final answer.
        give prority to latest date provided in metadata while answering user query.
        
        Using only the provided News Articles and chat history, respond to the user's inquiries in detail without omitting any context. 
        Provide relevant answers to the user's queries, adhering strictly to the content of the given articles and chat history.
        Dont start answer with based on . Dont provide extra information just provide answer what user asked.
        Answer should be very detailed in point format and preicise ,answer only based on user query and news articles.
        **If You cant find answer in provided articles dont make up answer on your own**
        *IF USER QUESTION IS ABOUT BUY/SELL THEN FORMULATE NICE ANSWER CONSISTS OF THIS -'Frruit is an AI-powered capital markets search engine designed to help you search , discover and analyze market data to make better-informed decisions. However, Frruit does not provide personalized investment advice or recommendations to buy or sell specific securities. For tailored investment guidance, please consult with a licensed financial advisor(USE THIS TEXT ONLY WHEN USER ASKING ABOUT BUY/SELL QUESTIONS).'AND PROVIDE DATA YOU HAVE SO THAT USER CAN DECIDE*

        **DONT PROVIDE ANY EXTRA INFORMATION APART FROM USER QUESTION AND ANSWER SHOULD BE IN PROPER MARKDOWN FORMATTING**
        
        The user has asked the following question: {input}


        
        """

        #R_prompt = PromptTemplate(template=res_prompt, input_variables=["context","input","date","cmots"])
        question_prompt = PromptTemplate(
        input_variables=["history", "cmots", "date","input"], template=res_prompt
        )
        #formatted_prompt = question_prompt.format(history=his, cmots=docs,date=date,input=memory_query)
        #entire_data=formatted_prompt

        

        #count=count_tokens(entire_data)/1000+0.5



        #print(count)
        #user_count=get_user_credits(user_id)
        # if count>user_count:
        #     raise HTTPException(status_code=400, detail="You have low credits! Please purchase a plan to continue accessing Frruit")

        #llm_chain_res= LLMChain(prompt=R_prompt, llm=llm1)
        #llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",streaming=True,stream_usage=True)

        ans_chain=question_prompt | llm_stream


    async def generate_chat_res(docs,history,input,date):
        if valid == 0:
            # Stream "Not a valid question" if the query is invalid
            
            error_message = "The search query you're trying to use does not appear to be related to the Indian financial markets. Please ensure your query focuses on topics like finance, investing, stocks, or other related subjects to maintain platform quality. If your query is relevant but isn’t yielding the desired results, consider refining it to improve accuracy and alignment with financial market topics"
            # Split the error message into smaller chunks
            error_chunks = error_message.split('. ')  # You can choose a different delimiter if needed
            
            for chunk in error_chunks:
                yield chunk.encode("utf-8")  # Yield each chunk as bytes
                await asyncio.sleep(1)  # Adjust the delay as needed for a smoother stream
                
            return
        

        aggregate = None
        #yield "Data:"+json.dumps(d)
        #yield "Response:"
        async for chunk in ans_chain.astream({"cmots": docs, "history": history, "input": input,"date":date}):
            #return chunk
            
            if chunk is not None:
                #print(chunk)
                answer = chunk.content
                aggregate = chunk if aggregate is None else aggregate + chunk
                if answer is not None:
                    await asyncio.sleep(0.01) 
                    yield answer.encode("utf-8")
                else:
                    pass
            else:
                print("Received None chunk")

      
        token_data=aggregate.usage_metadata
        total_tokens=token_data['total_tokens']/1000
        #print(total_tokens)

        #store_into_userplan(user_id,total_tokens)
        await insert_credit_usage(user_id,plan_id,total_tokens)

        links_data = {
            "links": []
        }
        combined_data = {**links_data}
        #print(combined_data)
        await store_into_db(session_id,prompt_history_id,combined_data)
    
        history = PostgresChatMessageHistory(
        connection_string=psql_url,
        session_id=session_id,
        )



        history.add_user_message(memory_query)
        history.add_ai_message(aggregate)


    #return "hello"
    return StreamingResponse(generate_chat_res(docs,his,memory_query,date), media_type="text/event-stream")


@web_rag.post("/web_rag")
async def web_rag_mix(
    request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...),
    ai_key_auth: str = Depends(authenticate_ai_key)
):
    query = request.query 
    valid,v_tokens,m_chat,h_chat=await query_validate(query,session_id)
    
    #print(valid)
    if valid == 0:
        #pass
        matched_docs,docs,memory_query,t_day,his= '', '', '', ''  ,'' 
    else:
        memory_query=await memory_chain(query,m_chat)
        #print(memory_query)
        date,user_q,t_day=await llm_get_date(memory_query)
        # bing,cmots=set_ret()

        # if date == "None":
        #     retriever_cmots = vs.as_retriever(search_kwargs={"k": 10})
        

        # else:
        #     retriever_cmots = vs.as_retriever(search_kwargs={"k": 20, 
        #                                                 'filter': {'date':{'$gte': int(date)}}
        #                                                }
        #                                 )
        
        docs, df ,links=await get_bing_results(memory_query)
        #print(docs[0])
        if docs is not None and df is not None:
            await insert_post1(df)
            #pinecone_task = asyncio.create_task(data_into_pinecone(df))
        else:
            pass
            #pinecone_task = None

        # # print(links)
        # docs= '\n'.join(docs)
        
        try:
            if date == 'None':
                results = vs.similarity_search_with_score(
                    memory_query,
                    k=10,

                )
                matched_docs=[doc[0].page_content for doc in results]
            #docs = retriever.invoke(memory_query)
            else:
                results = vs.similarity_search_with_score(
                    memory_query,
                    k=10,
                    filter={"date": {"$gte": int(date)}},

                )
                matched_docs=[doc[0].page_content for doc in results]
            #matched_docs = retriever_cmots.invoke(memory_query)
        except Exception as e:
            matched_docs = None  # or [] if you prefer an empty list
            # Optionally, log the exception or print it for debugging
            print(f"An error occurred: {e}")


        # for doc in matched_docs:
        #     docs += doc.page_content + "\n" 

        #print(docs)
        # count=count_tokens(docs)
        # print(count)
        
        
        #his,his_c=get_session_history(str(session_id))
        his=h_chat
        #print(his_c)
        # print(count_tokens(his))

        res_prompt = """
        Cmots Articles : {cmots}
        News Articles : {bing}
        chat history : {history}
        Today date:{date}
        You are a stock news and stock market information bot. 
        
        use the date provided in the metadata to answer the user query if the user is asking in specific time periods.
        If the same question {input} present in chat_history ignore that answer present in chat history dont consider that answer while generating final answer.
        give prority to latest date provided in metadata while answering user query.
        
        Using only the provided News Articles chat histor, respond to the user's inquiries in detail without omitting any context. 
        Provide relevant answers to the user's queries, adhering strictly to the content of the given articles and chat history.
        Dont start answer with based on . Dont provide extra information just provide answer what user asked.
        Answer should be very detailed in point format and preicise and dont provide any links ,answer only based on user query and news articles#.IN PROPER markdown formating.
        If You cant find answer in provided articles dont make up answer on your own.
        *IF USER QUESTION IS ABOUT BUY/SELL THEN FORMULATE NICE ANSWER CONSISTS OF THIS -'Frruit is an AI-powered capital markets search engine designed to help you search , discover and analyze market data to make better-informed decisions. However, Frruit does not provide personalized investment advice or recommendations to buy or sell specific securities. For tailored investment guidance, please consult with a licensed financial advisor(USE THIS TEXT ONLY WHEN USER ASKING ABOUT BUY/SELL QUESTIONS).'AND PROVIDE DATA YOU HAVE SO THAT USER CAN DECIDE*

        **DONT PROVIDE ANY EXTRA INFORMATION APART FROM USER QUESTION AND ANSWER SHOULD BE IN PROPER MARKDOWN FORMATTING**
        
        The user has asked the following question: {input}



        
        """

        R_prompt = PromptTemplate(template=res_prompt, input_variables=["cmots","bing","input","date","history"])
        #llm_chain_res= LLMChain(prompt=R_prompt, llm=llm1)
        #llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",streaming=True,stream_usage=True)
        #formatted_prompt = R_prompt.format(history=his, cmots=matched_docs,bing=docs,date=date,input=memory_query)
        #entire_data=formatted_prompt
        # count=count_tokens(entire_data)/1000  +3.5
        # #print(count)
        # user_count=get_user_credits(user_id)
        # if count>user_count:
        #     raise HTTPException(status_code=400, detail="You have low credits! Please purchase a plan to continue accessing Frruit.")
    
        ans_chain=R_prompt | llm_stream

    #final_task = asyncio.to_thread(ans_chain.invoke, {"context": docs, "cmots": matched_docs, "input": query,"date":t_day})
    
    
    async def generate_chat_res(matched_docs,docs,query,t_day,history):
        if valid == 0:
            # Stream "Not a valid question" if the query is invalid
            
            error_message = "The search query you're trying to use does not appear to be related to the Indian financial markets. Please ensure your query focuses on topics like finance, investing, stocks, or other related subjects to maintain platform quality. If your query is relevant but isn’t yielding the desired results, consider refining it to improve accuracy and alignment with financial market topics"
            # Split the error message into smaller chunks
            error_chunks = error_message.split('. ')  # You can choose a different delimiter if needed
            
            for chunk in error_chunks:
                yield chunk.encode("utf-8")  # Yield each chunk as bytes
                await asyncio.sleep(1)  # Adjust the delay as needed for a smoother stream
                
            return
        
        aggregate = None
        async for chunk in ans_chain.astream({"cmots": matched_docs,"bing":docs,"input": query,"date":t_day,"history":his}):
            #return chunk
            
            if chunk is not None:
                #print(type(chunk))
                answer = chunk.content
                aggregate = chunk if aggregate is None else aggregate + chunk
                if answer is not None:
                    await asyncio.sleep(0.01) 
                    yield answer.encode("utf-8")
                else:
                    pass
            else:
                print("Received None chunk")

        #yield b"metadata: " + json.dumps(aggregate.usage_metadata).encode("utf-8")
        #print(aggregate)
        token_data=aggregate.usage_metadata
        total_tokens=token_data['total_tokens']/1000 +3 
        #print(total_tokens)

        #store_into_userplan(user_id,total_tokens)
        await insert_credit_usage(user_id,plan_id,total_tokens)

        links_data = {
            "links": links
        }
        combined_data = {**links_data}
        #print(combined_data)
        await store_into_db(session_id,prompt_history_id,combined_data)
        #return aggregate.usage_metadata
        #background_tasks.add_task(handle_metadata, aggregate.usage_metadata if aggregate else {})
        history = PostgresChatMessageHistory(
        connection_string=psql_url,
        session_id=session_id,
        )



        history.add_user_message(memory_query)
        history.add_ai_message(aggregate)


    #return "hello"
    return StreamingResponse(generate_chat_res(matched_docs,docs,memory_query,t_day,his), media_type="text/event-stream")


@red_rag.post("/reddit_rag")
async def red_rag_bing(
    request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...),
    ai_key_auth: str = Depends(authenticate_ai_key)
):
    query = request.query  
    valid,v_tokens,m_chat,h_chat=await query_validate(query,session_id)
    #print(valid)
    if valid == 2:
        #pass
        his,docs,query= '', '', ''
    else:
        query=await memory_chain(query,m_chat)
        sr=await fetch_search_red(query)
        docs,df,links=await process_search_red(sr)
        #print(df)
        await insert_red(df)
        #print(links)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            history = PostgresChatMessageHistory(
            connection_string = psql_url,
            session_id=session_id,
            )
        #memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2, chat_memory=history)
            return history.messages,history
        
        #his,his_c=get_session_history(str(session_id))
        his=h_chat

        res_prompt = """
        Reddit articles: {context}
        chat history : {history}
        If the same question {input} present in chat_history ignore that answer present in chat history dont consider that answer while generating final answer.
        You are a stock news and stock market information bot. 
        Using only the provided Reddit Articles, respond to the user's inquiries in detail without omitting any context. 
        Provide relevant answers to the user's queries, adhering strictly to the content of the given articles.
        Dont start answer with based on . Dont provide extra information just provide answer what user asked.
        Answer should be very detailed in point format and preicise and dont provide any links ,answer only based on user query and news articles#.IN PROPER markdown formating.
        If You cant find answer in provided articles dont make up answer on your own.
        *IF USER QUESTION IS ABOUT BUY/SELL THEN FORMULATE NICE ANSWER CONSISTS OF THIS -'Frruit is an AI-powered capital markets search engine designed to help you search , discover and analyze market data to make better-informed decisions. However, Frruit does not provide personalized investment advice or recommendations to buy or sell specific securities. For tailored investment guidance, please consult with a licensed financial advisor(USE THIS TEXT ONLY WHEN USER ASKING ABOUT BUY/SELL QUESTIONS).'AND PROVIDE DATA YOU HAVE SO THAT USER CAN DECIDE*

        **DONT PROVIDE ANY EXTRA INFORMATION APART FROM USER QUESTION AND ANSWER SHOULD BE IN PROPER MARKDOWN FORMATTING**
        

        The user has asked the following question: {input}        
        """

        R_prompt = PromptTemplate(template=res_prompt, input_variables=["history","context","input"])
        #llm_chain_res= LLMChain(prompt=R_prompt, llm=llm1)
        #llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",streaming=True,stream_usage=True)
        # formatted_prompt = R_prompt.format(history=his, context=docs,input=query)
        # entire_data=formatted_prompt
        # count=count_tokens(entire_data)/1000 +3.5
        # #print(count)
        # user_count=get_user_credits(user_id)
        # if count>user_count:
        #     raise HTTPException(status_code=400, detail="You have low credits! Please purchase a plan to continue accessing Frruit.")
    
        ans_chain=R_prompt | llm_stream

        #final_task = asyncio.to_thread(ans_chain.invoke, {"context": docs, "cmots": matched_docs, "input": query,"date":t_day})
    
    
    async def generate_chat_res(his,docs,query):
        if valid == 2:
            # Stream "Not a valid question" if the query is invalid
            
            error_message = "The search query you're trying to use does not appear to be related to the Indian financial markets. Please ensure your query focuses on topics like finance, investing, stocks, or other related subjects to maintain platform quality. If your query is relevant but isn’t yielding the desired results, consider refining it to improve accuracy and alignment with financial market topics"
            # Split the error message into smaller chunks
            error_chunks = error_message.split('. ')  # You can choose a different delimiter if needed
            
            for chunk in error_chunks:
                yield chunk.encode("utf-8")  # Yield each chunk as bytes
                await asyncio.sleep(1)  # Adjust the delay as needed for a smoother stream
                
            return
        

        aggregate = None
        async for chunk in ans_chain.astream({"history":his,"context": docs,"input": query}):
            #return chunk
            
            if chunk is not None:
                #print(type(chunk))
                answer = chunk.content
                aggregate = chunk if aggregate is None else aggregate + chunk
                if answer is not None:
                    await asyncio.sleep(0.01) 
                    yield answer.encode("utf-8")
                else:
                    pass
            else:
                print("Received None chunk")

        #yield b"metadata: " + json.dumps(aggregate.usage_metadata).encode("utf-8")
        #print(aggregate)
        token_data=aggregate.usage_metadata
        total_tokens=token_data['total_tokens']/1000 +3
        #print(total_tokens)
        # total_tokens=total_tokens+3
        # print(total_tokens)

        #store_into_userplan(user_id,total_tokens)
        await insert_credit_usage(user_id,plan_id,total_tokens)

        links_data = {
            "links": links
        }
        combined_data = {**links_data}
        #print(combined_data)
        await store_into_db(session_id,prompt_history_id,combined_data)
    
        #return aggregate.usage_metadata
        # #background_tasks.add_task(handle_metadata, aggregate.usage_metadata if aggregate else {})
        history = PostgresChatMessageHistory(
        connection_string=psql_url,
        session_id=session_id,
        )



        history.add_user_message(query)
        history.add_ai_message(aggregate)


    #return "hello"
    return StreamingResponse(generate_chat_res(his,docs,query), media_type="text/event-stream")


@yt_rag.post("/yt_rag")
async def yt_rag_bing(request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...),
    ai_key_auth: str = Depends(authenticate_ai_key)
):
    query = request.query 
    valid,v_tokens,m_chat,h_chat=await query_validate(query,session_id)
    #print(valid)
    if valid == 2:
        #pass
        his,data,query= '', '', ''
    #start_time = time.time()
    else:
        links =await get_yt_data_async(query)
        #print(links)
        data = await get_data(links)

        
        #his,his_c=get_session_history(str(session_id))
        his=h_chat

        #data="Reliance Industries News - Get the Latest Reliance Industries News, Announcements, Photos & Videos on The Economic Times. Stock Quotes: Get all stocks market quotes, company stocks price quotes in India. ... Sensex News: On Friday, the Sensex surged nearly 1,300 points, while the Nifty50 reached a record high, driven by broad-based buying despite concerns over the capital gains tax hike in the Budget. The total market capitalization of BSE-listed companies increased by Rs 7.16 lakh crore to ... Discover the Reliance Industries Stock Liveblog, your ultimate resource for real-time updates and insightful analysis on a prominent stock. Keep track of Reliance Industries with the latest details, including: Last traded price 3037.95, Market capitalization: 2055275.89, Volume: 2800552, Price-to-earnings ratio 29.91, Earnings per share 101.6. Our comprehensive coverage combines fundamental and technical indicators to provide you with a comprehensive view of Reliance Industries's performance. Get all latest & breaking news on Reliance Industries. Watch videos, top stories and articles on Reliance Industries at moneycontrol.com. Reliance Q1 Results Updates: Reliance Industries Ltd (RIL), the energy-to-telecom-to-retail conglomerate, posted a 5.4 per cent year-on-year (YoY) decline in its net profit to ₹15,138 crore ... Reliance Infrastructure News - Get the Latest Reliance Infrastructure News, Announcements, Photos & Videos on The Economic Times. Stock Quotes: Get all stocks market quotes, company stocks price quotes in India. Visit Economic Times to read on Indian companies quotes listed on BSE NSE Stock Exchanges & search share prices by market capitalisation. Reliance Jio Q1 result: Jio reported an average revenue per user (ARPU) at ₹181.70, consistent with the previous quarter, and slightly up from ₹180.50 year-on-year, while total subscribers have risen to 489.7 million from 481.8 million last quarter and 448.5 million year-on-year. Reliance Industries Ltd share price was Rs 3,017.85 at close on 26th Jul 2024. Reliance Industries Ltd share price was up by 1.18 % over the previous closing price of Rs 2,982.60. Reliance Industries Ltd share price trend: Last 1 Month: Reliance Industries Ltd share price moved down by 0.31 % on BSE. Last 3 Months: Reliance Industries Ltd share ... Reliance Industries share price. Reliance Industries is trading 1.18 % upper at Rs 3,017.85 as compared to its last closing price. Reliance Industries has been trading in the price range of 3,025. ... Reliance Industries Share Price Live Updates: The analyst recommendation trend is shown below with the current rating as Buy. The median price target is ₹3379.0, 11.68% higher than current market price. The lowest target price among analyst estimates is ₹2600.0. The highest target price among analyst estimates is ₹3786.0. Reliance Industries Share Price Live Updates: Today, Reliance Industries' stock price dropped by 0.5% to reach ₹3025.55, while its industry counterparts are showing mixed results. Oil & Natural Gas Corporation and Petronet LNG are experiencing declines, whereas Oil India and Hindustan Petroleum Corporation are witnessing an increase in their stock prices. In general, the benchmark indices Nifty and Sensex are up by 0.09% and 0.12% respectively. Reliance Industries Share Price Today Live: A decrease in futures price, combined with an increase in open interest for Reliance Industries, indicates the possibility of downward price movement in the near future. Traders may consider maintaining their short positions."
        prompt = """
        Given youtube transcripts {summaries}
        chat_history {history}
        If the same question {query} present in chat_history ignore that answer present in chat history dont consider that answer while generating final answer.
        Using only the provided youtube video transcripts, respond to the user's inquiries in detail without omitting any context. 
        Provide relevant answers to the user's queries, adhering strictly to the content of the given transcripts.
        Dont start answer with based on . Dont provide extra information just provide answer what user asked in very detailing way.IN PROPER markdown formating.
        If You cant find answer ,please dont make up answer on your own.
        *IF USER QUESTION IS ABOUT BUY/SELL THEN FORMULATE NICE ANSWER CONSISTS OF THIS -'Frruit is an AI-powered capital markets search engine designed to help you search , discover and analyze market data to make better-informed decisions. However, Frruit does not provide personalized investment advice or recommendations to buy or sell specific securities. For tailored investment guidance, please consult with a licensed financial advisor(USE THIS TEXT ONLY WHEN USER ASKING ABOUT BUY/SELL QUESTIONS).'AND PROVIDE DATA YOU HAVE SO THAT USER CAN DECIDE*

        **DONT PROVIDE ANY EXTRA INFORMATION APART FROM USER QUESTION AND ANSWER SHOULD BE IN PROPER MARKDOWN FORMATTING**
        
        The user has asked the following question: {query}

        """
        #llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",streaming=True,stream_usage=True)
        yt_prompt = PromptTemplate(template=prompt, input_variables=["history","query", "summaries"])
        formatted_prompt = yt_prompt.format(history=his, summaries=data,query=query)
        # entire_data=formatted_prompt
        # count=count_tokens(entire_data)/1000 +3.5
        # #print(count)
        # user_count=get_user_credits(user_id)
        # if count>user_count:
        #     raise HTTPException(status_code=400, detail="You have low credits! Please purchase a plan to continue accessing Frruit.")
        chain = yt_prompt | llm_stream
    
    async def generate_chat_res(his,data,query):
        if valid == 2:
            # Stream "Not a valid question" if the query is invalid
            
            error_message = "The search query you're trying to use does not appear to be related to the Indian financial markets. Please ensure your query focuses on topics like finance, investing, stocks, or other related subjects to maintain platform quality. If your query is relevant but isn’t yielding the desired results, consider refining it to improve accuracy and alignment with financial market topics"
            # Split the error message into smaller chunks
            error_chunks = error_message.split('. ')  # You can choose a different delimiter if needed
            
            for chunk in error_chunks:
                yield chunk.encode("utf-8")  # Yield each chunk as bytes
                await asyncio.sleep(1)  # Adjust the delay as needed for a smoother stream
                
            return
        
        aggregate = None
        #yield "Data:"+json.dumps(d)
        #yield "Response:"
        async for chunk in chain.astream({"history":his,"summaries": data, "query": query}):
            #return chunk
            
            if chunk is not None:
                #print(chunk)
                #print(type(chunk))
                answer = chunk.content
                aggregate = chunk if aggregate is None else aggregate + chunk
                if answer is not None:
                    await asyncio.sleep(0.01) 
                    yield answer.encode("utf-8")
                else:
                    pass
            else:
                print("Received None chunk")

        token_data=aggregate.usage_metadata
        total_tokens=token_data['total_tokens']/1000 +3
        #print(total_tokens)

        #store_into_userplan(user_id,total_tokens)
        await insert_credit_usage(user_id,plan_id,total_tokens)

        links_data = {
            "links": links
        }
        combined_data = {**links_data}
        #print(combined_data)
        await store_into_db(session_id,prompt_history_id,combined_data)
    
        #return aggregate.usage_metadata
        # #background_tasks.add_task(handle_metadata, aggregate.usage_metadata if aggregate else {})
        history = PostgresChatMessageHistory(
        connection_string=psql_url,
        session_id=session_id,
        )



        history.add_user_message(query)
        history.add_ai_message(aggregate)


    #return "hello"
    return StreamingResponse(generate_chat_res(his,data,query), media_type="text/event-stream")