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

if os.getenv("OPENAI_API_TYPE") == "azure":
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")
    azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    
    if not all([azure_endpoint, api_key, api_version, azure_chat_deployment, azure_embedding_deployment]):
        raise ValueError("Azure OpenAI credentials are not fully set in the environment variables.")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    
    llm_kwargs = {
        "azure_endpoint": azure_endpoint,
        "api_key": api_key,
        "api_version": api_version,
        "azure_deployment": azure_chat_deployment,
    }
    
    GPT4o_mini = AzureChatOpenAI(temperature=0.2, **llm_kwargs)
    
else:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    GPT4o_mini = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")


client=chroma_server_client
vs= Chroma(
    client=client,
    collection_name="cmots_news",
    embedding_function=embeddings,)