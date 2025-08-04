import chromadb
from chromadb.config import Settings,DEFAULT_DATABASE,DEFAULT_TENANT
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv(override=True)


chroma_username=os.getenv("CHROMA_USERNAME")
chroma_password=os.getenv("CHROMA_PASSWORD")
chroma_host=os.getenv("CHROMA_HOST")

# print(f"{chroma_username}:{chroma_password}")


chroma_server_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=9000,
    # ... other settings
)


client=chroma_server_client
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs= Chroma(
    client=client,
    collection_name="cmots_news",
    embedding_function=embeddings,)



GPT3_4k = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
GPT3_16k = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
GPT4 = ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09")
GPT4o =ChatOpenAI(temperature=0, model="gpt-4o")
GPT4o_mini=ChatOpenAI(temperature=0.2, model="gpt-4o-mini")