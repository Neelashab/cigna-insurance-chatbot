from pinecone import Pinecone
from openai import OpenAI
from pydantic import BaseModel
from pydantic import BaseModel
from dotenv import load_dotenv
import os

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

class SmartQueries(BaseModel):
    queryDB: bool
    queries: list[str]

class ChatResponse(BaseModel):
    follow_up: bool
    response: str

def rewrite_query(user_query, client, currentSession: SessionState):
    prompt = f"""
    Identity: You are an intelligent system that extracts the most relevant search queries from a history of user input about health insurance for use in a vector database search.
    Instructions: Analyze the user's message to identify their main intent and specific information needs regarding health insurance (such as coverage, costs, enrollment, eligibility, providers, plan types, etc.).
    First, decide if it is true that the user prompt neccesitates a database query. 
    If false, return an empty list of queries. 
    If true, break down the user's request into distinct, actionable search queries that can be used to retrieve relevant information from a vector database.
    Consider the full conversation history for relevant information when generating queries, not just the last message.
    If two semantically different concepts are present in the user's message, create both separate queries for each concept and a combined query. 
    Output only the most relevant search queries as comma separated strings (e.g., 'health insurance coverage options', 'enrollment process for health insurance in Texas').
    Do not answer the user's question or include extra information; only provide the list of search queries.
    User input: {user_query}
    Take into account all past user queries for context: 
    {currentSession.all_input}
    Also take into account the last bot response for context:
    {currentSession.last_bot_response}
    Respond in the following format:
    queryDB: true or false
    queries: ["comma-separated list of search queries"]
    """
    print("")
    raw_response = client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.session_id,
        text_format=SmartQueries)
        #previous_response_id=currentSession.last_response_id)
    
    parsed = raw_response.output_parsed
    print("QUERY? YES OR NO: ", parsed.queryDB)
    print("QUERY SUGGESTIONS: ", parsed.queries)

    if not parsed.queryDB:
        print("No queries needed, returning empty list.")
    return parsed.queries if parsed.queryDB else []


# cache certain queries in the session state, if the same one comes up just reuse that? 
# is this even going to be useful realistically? 
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

    global last_response_id
    context = ""
    
    queries = rewrite_query(user_query, client, currentSession)

    if queries: 
        context += query_db(queries, top_k)

    # Prompt GPT for a response, injecting relevant context
    prompt = f"""You are a helpful assistant for existing Cigna customers or prospective new customers looking to buy insurance.
                First, decide if a follow up question is needed to clarify the user's intent, or if you can provide a direct answer based on the context provided.
                If a follow up question is needed, respond with a follow up question that will help you better understand the user's needs.
                If not, provide a direct answer based on relevant context, if provided.
                Context:
                {context}

                Question: {user_query}
                Respond in the following format:
                    follow_up: true or false
                    response: 'response or follow up question'"""

    raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.session_id,
        previous_response_id=(currentSession.last_response_id),
        text_format=ChatResponse)

    currentSession.last_response_id = raw_response.id
    parsed = raw_response.output_parsed

    print("A FOLLOW UP REQUIRED? ", parsed.follow_up)

    return parsed.response

# Chat loop
if __name__ == "__main__":
    currentSession = SessionState()
    print("Hello, welcome to Cigna Health Insurance! How can I help you?")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        currentSession.all_input += query + "\n"
        answer = ask_rag_bot(query, currentSession)
        currentSession.last_bot_response = answer

        print("\nAssistant:", answer)
