import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import json
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
from api.news_rag.bing_news import get_bing_results,insert_post,insert_post1,data_into_pinecone
from config import chroma_server_client
from langchain_chroma import Chroma




from dotenv import load_dotenv
load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY')
pg_ip=os.getenv('PG_IP_ADDRESS')
pine_api=os.getenv('PINECONE_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')
psql_url=os.getenv('DATABASE_URL')

# from langchain.globals import set_debug

# set_debug(True)



# index_name = "news"
# demo_namespace='newsrag'
index_name = "newsrag11052024"
demo_namespace='news'
embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

client=chroma_server_client
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs= Chroma(
    client=client,
    collection_name="cmots_news",
    embedding_function=embeddings,)



pc = PineconeClient(
 api_key=pine_api
)


index = pc.Index(index_name)

#demo_namespace='newsrag'
docsearch1 = Pinecone(
    index, embeddings, "text", namespace=demo_namespace
)


llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")#gpt-3.5-turbo-1106,gpt-3.5-turbo-16k
#llm1=ChatOpenAI(temperature=0.5, model="gpt-4o-2024-05-13")
llm_date = ChatOpenAI(temperature=0.5, model="gpt-4o-2024-05-13") #gpt-3.5-turbo-instruct,gpt-4-turbo,gpt-4o-2024-05-13
#llm1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro")

   
llama3 = ChatGroq(
    temperature=0.2,
    model="llama3-70b-8192",
    # api_key="" # Optional if not set as an environment variable
)
llama3_8b = ChatGroq(
    temperature=0,
    model="llama3-8b-8192",
)

def llm_get_date_lama(user_query):
    today = datetime.now().strftime("%Y-%m-%d")
    #print(today)
    date_prompt = """
        Today's date is {today}.
        Here's the user query: {user_query}
        Using the above given date for context, figure out which date the user wants data for.
        If the user query mentions "today" ,then use the above given today's date. Output the date in the format YYYYMMDD.
        If the user is asking about latest news in user query .output 2 days back date from todays date in YYYYMMDD.
        If the user is asking about recently news in user query .output 5 days back date from todays date in YYYYMMDD.
        If the user is aksing about specifc time period in user query. output the start date the user mentioned in YYYYMMDD format.
        If the user does not mention any date in user query outpute date as  "None" and If the user mention anything about quater and year output date as "None".

        Also, remove time-related references from the user query, ensuring it remains grammatically correct.

        Format your output as just as this DO NOT RETURN ANYTHING ELSE:
        YYYYMMDD,modified_user_query

        """
    query=user_query
    D_prompt = PromptTemplate(template=date_prompt, input_variables=["today","user_query"])
    llm_chain_date= LLMChain(prompt=D_prompt, llm=llama3)
    response=llm_chain_date.predict(today=today,user_query=query)
    #print(llm_get_date)
    response = response.strip()
    #print(response)
    date, general_user_query = split_input(response)

    return date,general_user_query

def llm_get_date(user_query):
    today = datetime.now().strftime("%Y-%m-%d")
    #print(today)
    date_prompt = """
        Today's date is {today}.
        Here's the user query: {user_query}
        Using the above given date for context, figure out which date the user wants data for.
        If the user query mentions "today" or "latest" then use the above given today's date. Output the date in the format YYYYMMDD.
        If the user is asking about recently news in user query .output 7 days back date from todays date in YYYYMMDD.
        If the user is aksing about specifc time period in user query from past. output the start date the user mentioned in YYYYMMDD format.
        If the user doesnot mention any date in user query or asking about upcoming date outpute date as  "None" and If the user mention anything about quater and year output date as "None".

        Also, remove time-related references from the user query, ensuring it remains grammatically correct.

        Format your output as:
        YYYYMMDD,modified_user_query

        """
    query=user_query
    D_prompt = PromptTemplate(template=date_prompt, input_variables=["today","user_query"])
    llm_chain_date= LLMChain(prompt=D_prompt, llm=llm_date)
    response=llm_chain_date.predict(today=today,user_query=query)
    response = response.strip()
    print(response)
    date, general_user_query = split_input(response)

    return date,general_user_query,today


def split_input(input_string):
    # Split the input string at the first comma
    parts = input_string.split(',', 1)
    # Assign the parts to date and general_user_query
    date = parts[0].strip()
    general_user_query = parts[1].strip() if len(parts) > 1 else ""
    return date, general_user_query


def cmots_only(query,session_id):
    memory_query=memory_chain(query,session_id)
    date,user_q,t_day=llm_get_date(memory_query)
    print(date)
    user_q=memory_query
    text_field = "text"  
    vectorstore = PineconeVectorStore(  
        index, embeddings, text_field  ,namespace=demo_namespace
    )  
    if date == "None":
        retriever = vs.as_retriever(search_kwargs={"k": 20}
                                    )
    else:
        retriever = vs.as_retriever(search_kwargs={"k": 20, 
                                                    'filter': {'date':{'$gte': int(date)}}
                                                   }
                                    )

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history use only last two memories and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and if the latest question doesnt have any relevancy to chat history  return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    #print(contextualize_q_prompt)
    history_aware_retriever = create_history_aware_retriever(
        llm1, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt = """You are a stock news and stock market information bot. 
        Todays date is {date}
        Using only the provided context, respond to the user's inquiries in detail without omitting any context. 
        Dont browse internet if you dont have any context.
        Provide relevant answers to the user's queries, adhering strictly to the content of the given articles.
        Dont start answer with based on . Dont provide extra information just provide answer for what user asked.
        use provided date as reference if user asking about future or upcoming events.
        If You cant find answer in provided context dont make up answer on your own.


    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm1, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        history = PostgresChatMessageHistory(
        connection_string = psql_url,
        session_id=session_id,
        )
       #memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2, chat_memory=history)
        return history


    
    
    #memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2, chat_memory=history)


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        #input_messages_key="date",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    # qa = RetrievalQA.from_chain_type(  
    #                             llm=llm1,  
    #                             chain_type="stuff",  
    #                             retriever=retriever
    #                             )  
    # res=qa.run(query)  
    with get_openai_callback() as cb:
        resp=conversational_rag_chain.invoke(
            {"input": query,"date":t_day},
            config={"configurable": {"session_id": session_id}},
        )["answer"] 
        #resp=rag_chain.invoke({"input": query, "chat_history": get_session_history})
        # print(f"Total Tokens: {cb.total_tokens}")
        # print(f"Prompt Tokens: {cb.prompt_tokens}")
        # print(f"Completion Tokens: {cb.completion_tokens}")
        # print(f"Total Cost (USD): ${cb.total_cost}")

    return {"Response": resp,
            "links":[],
            "Total_Tokens": cb.total_tokens,
            "Prompt_Tokens": cb.prompt_tokens,
            "Completion_Tokens": cb.completion_tokens,
            #"Total Cost (USD)": cb.total_cost
            }


def memory_chain(query,session_id):
    connection = psycopg2.connect(psql_url)

    # Create a cursor
    cursor = connection.cursor()

    # View the table
    table_name = 'message_store'
    s_id=session_id
    cursor.execute("SELECT message FROM message_store WHERE session_id = %s", (session_id,))
    rows = cursor.fetchall()

    for row in rows:
        print(row)

    # Close the cursor and connection
    cursor.close()
    connection.close()


    chat=rows[:6]
    contextualize_q_system_prompt = """Given a chat history {chat_his}and the latest user question {query}\
    which might reference context in the chat history, formulate a standalone question if needed include time/date part also based on user previous question.\
    which can be understood without the chat history. Do NOT answer the question, \
    if the latest question {query} doesnot have any relevancy to chat history return it as is with out changing user query: {query}.
    
    """

    c_q_prompt=PromptTemplate(template=contextualize_q_system_prompt,input_variables=['chat_his','query'])
    llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18",temperature=0.5)
    memory_chain=LLMChain(prompt=c_q_prompt,llm=llm)
    res=memory_chain.predict(query=query,chat_his=chat)

    return res

pc = PineconeClient(
 api_key=pine_api
)

def set_ret():
    embeddings = OpenAIEmbeddings()
    index_name = "newsrag11052024"
    demo_namespace='news'

    index = pc.Index(index_name)

    #demo_namespace='newsrag'
    docsearch_cmots = Pinecone(
        index, embeddings, "text", namespace=demo_namespace
    )

    index_name1 = "bing-news"
    demo_namespace1='bing'

    index1 = pc.Index(index_name1)
    embeddings = OpenAIEmbeddings()

    #demo_namespace='newsrag'
    docsearch_bing = Pinecone(
        index1, embeddings, "text",namespace=demo_namespace1
    )

    return docsearch_bing,docsearch_cmots


# app = FastAPI()

# class SearchRequest(BaseModel):
#     query: str
#     session_id: str




# @app.post("/new_rag")
def cmots_bing_Search(query,session_id):
    memory_query=memory_chain(query,session_id)
    print(memory_query)
    #pine=get_stock_news_summary(query)
    #print(pine)
    date,user_q=llm_get_date_lama(memory_query)
    print(date,user_q)
    bing,cmots=set_ret()
    #retriever_bing = bing.as_retriever(search_kwargs={"k": 15})

    if date == "None":
        retriever_cmots = cmots.as_retriever(search_kwargs={"k": 10})
        retriever_bing = bing.as_retriever(search_kwargs={"k": 15})

    else:
        retriever_cmots = cmots.as_retriever(search_kwargs={"k": 20, 
                                                    'filter': {'date':{'$gte': int(date)}}
                                                   }
                                    )
    
        retriever_bing = bing.as_retriever(search_kwargs={"k": 10,'filter': {'date':{'$gte': int(date)}}
                                                   }
                                    )

    lotr = MergerRetriever(retrievers=[retriever_cmots , retriever_bing])
    matched_docs = lotr.invoke(memory_query)
    # print(matched_docs)
    res_prompt = """
    News Articles : {context}
    You are a stock news and stock market information bot. 
    
    use the date provided in the metadata to answer the user query if the user is asking in specific time periods.
    give prority to latest date provided in metadata while answering user query.
    
    Using only the provided News Articles, respond to the user's inquiries in detail without omitting any context. 
    Provide relevant answers to the user's queries, adhering strictly to the content of the given articles.
    Dont start answer with based on . Dont provide extra information just provide answer what user asked.

    The user has asked the following question: {input}

    The output should contain generated answer and news urls that used while generating the answer.

    The output should be in json format:
    "answer": Answer should be very detailed in point format and preicise ,answer only based on user query and news articles.Dont include links here.
    "links": list urls which are used in generating answer.If no links are related to user query return empty.
    """

    R_prompt = PromptTemplate(template=res_prompt, input_variables=["context","input"])
    #llm_chain_res= LLMChain(prompt=R_prompt, llm=llm1)
    #resp=llm_chain_res.predict(today=today,relevant_articles=relevant_articles,general_user_query=query)
    output_parser=JsonOutputParser()
    ans_chain=R_prompt | llm1 | output_parser
    #retriever = lotr
    # R_prompt =PromptTemplate.from_template(template=res_prompt)
    # combine_docs_chain = create_stuff_documents_chain(
    # llm_date, R_prompt
    # )
    # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


    # def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #     history = PostgresChatMessageHistory(
    
    #     session_id=session_id,
    #     )
    #     return history

    # conversational_rag_chain = RunnableWithMessageHistory(
    #     ans_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
    
    # with get_openai_callback() as cb:
        # resp=conversational_rag_chain.invoke(
        #     {"relevant_articles":matched_docs,"general_user_query":query},
        #     config={"configurable": {"session_id": session_id}},
        # )
    
    #resp=rag_chain.invoke({"input": query, "chat_history": get_session_history})
    # print(f"Total Tokens: {cb.total_tokens}")
    # print(f"Prompt Tokens: {cb.prompt_tokens}")
    # print(f"Completion Tokens: {cb.completion_tokens}")
    # print(f"Total Cost (USD): ${cb.total_cost}")

    # return {"Response": resp,
    #         "Total_Tokens": cb.total_tokens,
    #         "Prompt_Tokens": cb.prompt_tokens,
    #         "Completion_Tokens": cb.completion_tokens,
    #         #"Total Cost (USD)": cb.total_cost
    #         }

    history = PostgresChatMessageHistory(
    connection_string=psql_url,
    session_id=session_id,
    )
    

    with get_openai_callback() as cb:
          final=ans_chain.invoke({"context":matched_docs,"input":user_q})
    #     final = retrieval_chain.invoke({"input": memory_query})['answer']
    print(final)

    # data = json.loads(final)

    # # Extract answer and links
    # answer = data['answer']
    # links = data['links']

    #print(final)
    history.add_user_message(memory_query)
    history.add_ai_message(final['answer'])
    return{"Response": final['answer'],
            "links":final['links'],
            "Total_Tokens": cb.total_tokens,
            "Prompt_Tokens": cb.prompt_tokens,
            "Completion_Tokens": cb.completion_tokens,
            #"Total Cost (USD)": cb.total_cost
            }


    
    # retriever_cmots = cmots.as_retriever(search_kwargs={"k": 10})


async def web_rag(query,session_id):
    memory_query=memory_chain(query,session_id)
    print(memory_query)
    date,user_q,t_day=llm_get_date(memory_query)
    #bing,cmots=set_ret()

    if date == "None":
        retriever_cmots = vs.as_retriever(search_kwargs={"k": 10})
    

    else:
        retriever_cmots = vs.as_retriever(search_kwargs={"k": 20, 
                                                    'filter': {'date':{'$gte': int(date)}}
                                                   }
                                    )
    
    docs, df = get_bing_results(query)
    #print(docs)
    if docs is not None and df is not None:
        insert_post1(df)
        pinecone_task = asyncio.create_task(data_into_pinecone(df))
    else:
        pinecone_task = None

    #start_time1= time.time()
    matched_docs = retriever_cmots.invoke(memory_query)
    #end_time1 = time.time()

    #execution_time = end_time1 - start_time1
    #print(f"Execution Time for invoke: {execution_time} seconds")
    #lotr = MergerRetriever(retrievers=[retriever_cmots , retriever_bing])
    #matched_docs = lotr.invoke(memory_query)

    res_prompt = """
    News Articles : {context}
    cmots news articles :{cmots}
    Today date:{date}
    You are a stock news and stock market information bot. 
    
    use the date provided in the metadata to answer the user query if the user is asking in specific time periods.
    give prority to latest date provided in metadata while answering user query.
    
    Using only the provided News Articles, respond to the user's inquiries in detail without omitting any context. 
    Provide relevant answers to the user's queries, adhering strictly to the content of the given articles.
    Dont start answer with based on . Dont provide extra information just provide answer what user asked.
    If You cant find answer in provided articles dont make up answer on your own.

    The user has asked the following question: {input}


    The output should contain generated answer and news urls that used while generating the answer.

    The output should be in json format:
    ** Dont provide in list format in answer and links **
    "answer": Answer should be very detailed in point format and preicise ,answer only based on user query and news articles.Dont include links here.should be a string not a list.IN PROPER markdown formating.
    "links": list urls which are used in generating answer.If no links are related to user query return empty.
    """

    R_prompt = PromptTemplate(template=res_prompt, input_variables=["context","input","date","cmots"])
    #llm_chain_res= LLMChain(prompt=R_prompt, llm=llm1)
    llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
    #resp=llm_chain_res.predict(today=today,relevant_articles=relevant_articles,general_user_query=query)
    output_parser=JsonOutputParser()
    ans_chain=R_prompt | llm1 | output_parser

    #final=ans_chain.invoke({"context":docs,"context2":tdocs,"cmots":matched_docs,"input":query} )
    final_task = asyncio.to_thread(ans_chain.invoke, {"context": docs, "cmots": matched_docs, "input": query,"date":t_day})

  
    with get_openai_callback() as cb:
        if pinecone_task:
            final, _ = await asyncio.gather(final_task, pinecone_task)
        else:
            final = await final_task

    history = PostgresChatMessageHistory(
    connection_string=psql_url,
    session_id=session_id,
    )

    history.add_user_message(memory_query)
    history.add_ai_message(final['answer'])
    return{"Response": final['answer'],
            "links":final['links'],
            "Total_Tokens": cb.total_tokens,
            "Prompt_Tokens": cb.prompt_tokens,
            "Completion_Tokens": cb.completion_tokens,
            #"Total Cost (USD)": cb.total_cost
            }

    # return final



# import time
# import asyncio

# start_time = time.time()
# res = asyncio.run(web_rag("what is the latest news on bajaj finserv", "2607okkklmpp"))
# #res= cmots_only("","2607joj")
# end_time = time.time()

# execution_time = end_time - start_time
# print(f"Execution Time: {execution_time} seconds")
# print(res)