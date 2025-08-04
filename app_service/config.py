import chromadb
from chromadb.config import Settings,DEFAULT_DATABASE,DEFAULT_TENANT
import os
from langchain_openai import ChatOpenAI
import chromadb.utils.embedding_functions as embedding_functions
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv(override=True)

#CHROMA_SERVER
chroma_username=os.getenv("CHROMA_USERNAME")
chroma_password=os.getenv("CHROMA_PASSWORD")
chroma_host=os.getenv("CHROMA_HOST")

chroma_server_client= chromadb.HttpClient(
    host=chroma_host,
    port=9000,
    ssl=False,
    headers=None,
    settings=Settings(chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",chroma_client_auth_credentials=f"{chroma_username}:{chroma_password}", allow_reset=True),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)


client=chroma_server_client
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs= Chroma(
    client=client,
    collection_name="cmots_news",
    embedding_function=embeddings,)

vs_promoter= Chroma(
    client=client,
    collection_name="promoters_202409",  #promoters,promoters_202409
    embedding_function=embeddings,)



openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )


default_ef = embedding_functions.DefaultEmbeddingFunction()



GPT3_4k = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
GPT3_16k = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
GPT4 = ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09")
GPT4o =ChatOpenAI(temperature=0, model="gpt-4o")
GPT4o_mini=ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
llm_stream = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",stream_usage=True,streaming=True)
llm_date = ChatOpenAI(temperature=0.3, model="gpt-4o-2024-05-13")


llm_screener = ChatOpenAI(temperature = 0.5 ,model ='gpt-4o-mini')

#POSTGRES