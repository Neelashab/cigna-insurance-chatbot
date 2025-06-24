from pinecone import Pinecone
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NAMESPACE = os.getenv("NAMESPACE")

# Create client 
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_NAME)
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
    clarify: bool
    queryDB: bool
    queries: list[str]
    
class ChatResponse(BaseModel):
    response: str

def rewrite_query(user_query, client, currentSession: SessionState):
    prompt = f"""
    Identity: You are an intelligent system that extracts the most relevant search queries from a history of user input about health insurance for use in a vector database search.
    Instructions: Analyze the user's message to identify their main intent and specific information needs regarding health insurance (such as coverage, costs, enrollment, eligibility, providers, plan types, etc.).
    First, decide if it is true that the user prompt neccesitates a database query. 
    If false, return an empty list of queries. 
    Then, determine whether the user query has vague terms or intent. For example, terms such as "my state" or "injury" may not be specific enough to create the most accurate query. 
    If true, also return an empty list of queries. 
    If a query is required and there is no need for a clarification, break down the user's request into distinct, actionable search queries that can be used to retrieve relevant information from a vector database.
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
    clarify: true or false
    queries: ["comma-separated list of search queries"]
    """
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


# TODO implement semantic caching
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

    # Prompt GPT for a response, injecting relevant context
    prompt = f"""You are a helpful assistant for existing Cigna customers or prospective new customers looking to buy insurance.
                First, determine whether it is necessary to clarify any vague terms or intent in the user's request. 
                For example, if a user asks about "my state" or "injury", you may need to ask a clarifying question.
                Clarification required: {query_analysis.clarify}
                
                If true, respond with a clarifying question that will help you better understand the user's needs.
                If not, provide a direct answer based on relevant context, if provided.
                Context:
                {context}

                Question: {user_query}
                Respond in the following format:
                    response: 'response or follow up question'"""

    raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.session_id,
        previous_response_id=(currentSession.last_response_id),
        text_format=ChatResponse)

    currentSession.last_response_id = raw_response.id
    parsed = raw_response.output_parsed

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


