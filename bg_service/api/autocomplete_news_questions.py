from langchain_community.llms import HuggingFaceEndpoint
from fastapi import FastAPI,APIRouter
import os
import chromadb
import re
import requests
import json
import spacy
import chromadb.utils.embedding_functions as embedding_functions
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from starlette.status import HTTP_403_FORBIDDEN

load_dotenv(override=True)

router = APIRouter()

API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['CURL_CA_BUNDLE'] = ''
psql_url=os.getenv('DATABASE_URL')

# Database connection parameters
db_url = psql_url


def get_data_from_url(url, bearer_token):

    # Headers with Authorization
    headers = {"Authorization": f"Bearer {bearer_token}"}

    try:
        # Sending GET request with headers
        response = requests.get(url, headers=headers)

        # Checking if request was successful (status code 200)
        if response.status_code == 200:
            # Extracting JSON data
            data = response.json()

            articles = data["data"]

            if articles:
                for article in articles:
                    # Cleaning HTML tags from article text
                    clean_text = re.sub(r"<.*?>", "", article["arttext"])
                    article["arttext"] = clean_text

                return articles
            else:
                print("No articles found for today's date.")
                return []
        else:
            print("Failed to fetch data. Status code:", response.status_code)

    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)


def fetch_all_ids():
    try:
        # Connect to the database
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        # SQL query to fetch all IDs
        fetch_query = "SELECT serial_no FROM suggested_news_prompt"

        # Execute the query
        cursor.execute(fetch_query)

        # Fetch all results
        ids = cursor.fetchall()

        # Print the fetched IDs
        for id_tuple in ids:
            print(id_tuple[0])

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            cursor.close()
            conn.close()


def get_articles():
    url1 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/hot-pursuit/10"
    url2 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/stock-alert/10"
    url3 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/corporate-news/10"
    url4 = "http://airrchipapis.cmots.com/api/CapitalMarketLiveNews/market-beat/10"

    urls = [url1, url2, url3, url4]
    #urls=[url3]

    bearer_token = os.getenv("CMOTS_BEARER_TOKEN")

    #existing_ids = fetch_all_ids() if fetch_all_ids() else []
    existing_ids = []
    articles = []
    for url in urls:
        print("Fetching data from URL:", url)
        cleaned_articles = get_data_from_url(url, bearer_token)

        # Filter articles that don't already exist in ChromaDB
        new_articles = [ article for article in cleaned_articles if article["sno"] not in existing_ids
        ]
        print("Number of new articles:", len(new_articles))
        articles.extend(new_articles)

    # print(json.dumps(articles, indent = 4))
    #print(articles)
    return articles


API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
)
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def extract_organizations(text):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    # Process the text with spaCy
    doc = nlp(text)
    # Extract organization names
    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return organizations


def extract_question(text):
    # Use regex to find the last 'question: <question>' pattern
    matches = re.findall(r"question: \s*(?!<question>)(.*)", text)
    valid_questions = [match for match in matches if len(match.split()) > 3]
    if valid_questions:
        return valid_questions
    return []


def insert_questions_into_database(list_of_questions):
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO suggested_news_prompt (serial_no, question)
            VALUES (%s, %s)
        """
        
        for questions in list_of_questions:
            for question in questions["questions"]:
                cursor.execute(insert_query, (questions["sno"], question))

        # Commit the transaction
        conn.commit()
        print("Data inserted successfully")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            cursor.close()
            conn.close()


def insert_questions_into_database_all(list_of_questions):
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO suggested_news_prompt (serial_no, question)
            VALUES (%s, %s)
        """
        
        # Prepare the data for bulk insert
        data_to_insert = []
        for questions in list_of_questions:
            for question in questions["questions"]:
                data_to_insert.append((questions["sno"], question))

        # Use executemany to insert all records in a single call
        cursor.executemany(insert_query, data_to_insert)

        # Commit the transaction
        conn.commit()
        print("Data inserted successfully")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            cursor.close()
            conn.close()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=10000,
    do_sample=False,
    temperature=0.5
)



def generate_questions(articles):    
    list_of_json_objects = []
    
    for article in articles:
        #print(article)
        questions = []
        companies = extract_organizations(article["arttext"])
        print(companies)
        input_prompt = f"""
        data: {article["arttext"]}
        generate ONLY 10 complex, but short questions from the previously given data but make sure you cover all of these companies: {companies}. 
        The questions should not be repetitive, ask different kinds of questions.
        give output ONLY in this format and nothing else: 
        'question: <question>'
        'question: <question>'
        'question: <question>'
        
        ...and so on
        """
        data = query(
            {
                "inputs": input_prompt,
                "parameters": {
                    "max_new_tokens": 15000,
                    "top_p": 0.1,
                    "temperature": 0.8,
                },
            }
        )
        try:
            print(data[0]["generated_text"])
            print()
            questions.append(data[0]["generated_text"])
        except:
            print(data)
            print()
            questions.append(data)
        
        extracted_questions = []
        for question in questions:
            temp_questions = []
            temp_questions = extract_question(question)
            print(temp_questions)
            for temp in temp_questions:
                print(temp)
                extracted_questions.append(temp)
                
        json_object = {"sno": article["sno"], "questions": extracted_questions}
        list_of_json_objects.append(json_object)

    return list_of_json_objects

def generate_qs(articles):
    list_of_json_objects = []

    summary_template = ''' 
    News heading {heading}
    News article {article}.
    Read the news article above and generate three standalone questions , one question based on heading and two questions based on article.
    Return output ONLY in this format and nothing else: 
        'question: <question>'
        'question: <question>'
        'question: <question>'
        
        ...and so on

    '''
    summary_prompt = PromptTemplate(template=summary_template, input_variables=["heading","article"])
    # llm_summary = OpenAI(temperature=0.6)
    #llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini", max_tokens=1000)
    llm_chain_summary = LLMChain(prompt=summary_prompt, llm=llm)

    questions = []
    for article in articles:
        d=llm_chain_summary.invoke({"article":article['arttext'],"heading":article['heading']})
        # print(d['text'])
        questions.append(d['text'])
        extracted_questions = []
        for question in questions:
            temp_questions = []
            temp_questions = extract_question(question)
            #print(temp_questions)
            for temp in temp_questions:
                #print(temp)
                extracted_questions.append(temp)
                
        json_object = {"sno": article["sno"], "questions": extracted_questions}
        list_of_json_objects.append(json_object)
    #print(list_of_json_objects)

    return list_of_json_objects

AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )
    


@router.post("/get_news_questions")
def get_new_questions(ai_key_auth: str = Depends(authenticate_ai_key)):
    articles = get_articles()

    #questions = generate_questions(articles)
    questions = generate_qs(articles)

    #print("qs generated")
    insert_questions_into_database_all(questions)

    return questions

#get_new_questions()

