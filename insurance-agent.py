from pinecone import Pinecone
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal 
from datetime import date
import os

from schemas import PlanDiscoveryResponse

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE = os.getenv("NAMESPACE")

# Create client 
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)
client = OpenAI(api_key=OPENAI_API_KEY)




# Initialize the session state

class BusinessProfile(BaseModel):
    business_name: str
    business_size: Literal["2–99", "100–499", "500–2,999", "3,000+"]
    business_type: Literal[
        "Hospital and Health Systems", "Higher Education", "K-12 Education",
        "State and Local Governments", "Taft-Hartley and Federal",
        "Third Party Administrators (Payer Solutions)"
    ]
    state: str
    creation_date: date
    other: list[str]
    response: str

class SessionState:
    def __init__(self):
        self.last_response_id = None
        # for production generate a new one each time. keep this for testing purposes
        self.session_id = "development_id"

        # so far, just keeping track of past user queries and last bot response 
        # could also consider keeping track of full chat history 
        # or entity recognition on the chat history 
        self.all_input = ""
        self.last_bot_response = ""
        self.business_profile: BusinessProfile | None = None




class SmartQueries(BaseModel):
    clarify: bool
    queryDB: bool
    queries: list[str]
    
class ChatResponse(BaseModel):
    response: str

def rewrite_query(user_query, client, currentSession: SessionState):
    with open("prompts/rewrite_query.txt") as f:
        prompt = f.read()

    print("")
    raw_response = client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.session_id,
        text_format=SmartQueries)
    
    parsed = raw_response.output_parsed
    print("QUERY? YES OR NO: ", parsed.queryDB)
    print("CLARIFICATION QUESTION NEEDED? YES OR NO: ", parsed.clarify)
    print("QUERY SUGGESTIONS: ", parsed.queries)

    if not parsed.queryDB:
        print("No queries needed, returning empty list.")
    return parsed

def query_db(queries: list[str], top_k: int = 5):
    context = ""
    for query in queries:
            print("Searching for:", query)  
            results = index.search(
                namespace=NAMESPACE,
                query={
                    "inputs": {"text": query},
                    "top_k": top_k
                },
                rerank={
                    "model": "bge-reranker-v2-m3",
                    "top_n": top_k,
                    "rank_fields": ["chunk_text"]
                },
                fields=["chunk_text", "source"]
            )
            hits = results.get("result", {}).get("hits", [])
            context += "\n\n".join([hit["fields"].get("chunk_text", "") for hit in hits])
    
    return context

def ask_rag_bot(user_query: str,  currentSession: SessionState, top_k: int = 5):
    context = ""
    
    query_analysis = rewrite_query(user_query, client, currentSession)
    queries = query_analysis.queries

    if queries: 
        context += query_db(queries, top_k)
    
    print("Context retrieved from database: ", context)

    with open("prompts/rag_bot.txt") as f:
        prompt = f.read()

    raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.session_id,
        previous_response_id=(currentSession.last_response_id),
        text_format=ChatResponse)

    currentSession.last_response_id = raw_response.id
    parsed = raw_response.output_parsed

    return parsed.response


def get_business_info_node(user_query: str, currentSession: SessionState):
    """ This function is for determining information about the business' elgibility for certain plans. This information is constant and does not change
    during the session, thus it is stored as part of the state."""

    with open("prompts/rag_bot.txt") as f:
        prompt = f.read()

    raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.session_id,
        previous_response_id=(currentSession.last_response_id),
        text_format=BusinessProfile)

    currentSession.last_response_id = raw_response.id
    parsed = raw_response.output_parsed

    return parsed.response


# TODO 
# 1) Get 


def plan_discovery_node(user_query: str, currentSession: SessionState, eligibility_info: dict):
    """ This function is for determining information about the business' preferences for certain plans. 
    This information is subject to change during the session."""

    with open("prompts/get_business_info.txt") as f:
        prompt = f.read()

    raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.session_id,
        previous_response_id=(currentSession.last_response_id),
        text_format=PlanDiscoveryResponse)

    currentSession.last_response_id = raw_response.id
    parsed = raw_response.output_parsed

    return parsed.response



# Chat loop
if __name__ == "__main__":
    currentSession = SessionState()
    

    # Conversational Loop 

    print("Hello, welcome to Cigna Health Insurance! I am a helpful bot designed to help your company buy health insurance.")
    print("To begin, could you please tell me a bit about your business?")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        currentSession.all_input += query + "\n"

        
        response = get_business_info_node(query, currentSession)
        currentSession.last_bot_response = response

        print("\nAssistant:", response)


