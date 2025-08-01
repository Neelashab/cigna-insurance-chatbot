from pinecone import Pinecone
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import date
from typing import Literal 
import os
import tiktoken
import uuid
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from schemas import BusinessProfile, PlanDiscoveryResponse, PlanDiscoveryAnswers

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

# Initialize tokenizer and NER pipeline
tokenizer = tiktoken.encoding_for_model("gpt-4")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Informative links for reasoning model 
links = ["https://www.cigna.com/employers/medical-plans/",
         "https://www.cigna.com/knowledge-center/types-of-health-insurance",
         "https://www.cigna.com/individuals-families/shop-plans/plans-through-employer/",
         "https://www.cigna.com/knowledge-center/copays-deductibles-coinsurance",
         "https://www.cigna.com/knowledge-center/in-network-vs-out-of-network"]

# Initialize the session state
class SessionState:
    def __init__(self):
        self.user_id = str(uuid.uuid4())
        self.chat_history = []
        self.plan_discovery_answers: PlanDiscoveryAnswers | None = None
        self.extracted_entities = []
    
    def update_chat_history(self, role: Literal["user", "assistant"], content: str):
        self.chat_history.append({"role": role, "content": content})
        self.manage_token_limit()
    
    def count_tokens(self, messages):
        """Count tokens in a list of messages"""
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            total_tokens += len(tokenizer.encode(content))
        return total_tokens
    
    def extract_entities(self, text):
        """Extract entities from text using NER pipeline"""
        try:
            entities = ner_pipeline(text)
            # Filter and format entities
            formatted_entities = []
            for entity in entities:
                if entity['score'] > 0.9:  # High confidence threshold
                    formatted_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'score': entity['score']
                    })
            return formatted_entities
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
    
    def summarize_conversation_chunk(self, messages):
        """Summarize a chunk of conversation messages"""
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Extract entities before summarization
        entities = self.extract_entities(conversation_text)
        self.extracted_entities.extend(entities)
        
        # Create summarization prompt
        summary_prompt = f"""Summarize the following conversation while preserving key insurance-related information, preferences, and business details:

{conversation_text}

Provide a concise summary that maintains important context for insurance discussions."""
        
        try:
            response = client.responses.parse(
                model="gpt-4o-mini",
                input=[{"role": "system", "content": summary_prompt}],
                user=self.user_id,
                text_format=SummaryResponse
            )
            return response.output_parsed.summary
        except Exception as e:
            print(f"Summarization error: {e}")
            return f"Summary of {len(messages)} messages (summary failed)"
    
    def format_conversation_history(self):
        """Format conversation history for prompt inclusion"""
        if not self.chat_history:
            return ""
        
        return "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self.chat_history
        ])
    
    def format_extracted_entities(self, limit=10):
        """Format extracted entities for prompt inclusion"""
        if not self.extracted_entities:
            return 'None'
        
        recent_entities = self.extracted_entities[-limit:]
        return [e['text'] for e in recent_entities]
    
    def manage_token_limit(self, max_tokens=300, percent_to_summarize=0.2):
        """Manage token limit by summarizing older conversation history"""
        if not self.chat_history:
            return
        
        current_tokens = self.count_tokens(self.chat_history)
        
        if current_tokens <= max_tokens:
            return
        
        print("SUMMARIZING MESSAGES")
        
        # Calculate number of messages to summarize
        num_messages_to_summarize = int(len(self.chat_history) * percent_to_summarize)
        if num_messages_to_summarize < 2:
            num_messages_to_summarize = 2
        print("MESSAGES TO SUM UP: ", num_messages_to_summarize)
        
        # Get messages to summarize (oldest messages)
        messages_to_summarize = self.chat_history[:num_messages_to_summarize]
        remaining_messages = self.chat_history[num_messages_to_summarize:]
        
        # Create summary
        summary = self.summarize_conversation_chunk(messages_to_summarize)
        print("SUMMARY: ", summary)
        
        # Replace summarized messages with summary
        summary_message = {
            "role": "system", 
            "content": f"[CONVERSATION SUMMARY] {summary}"
        }
        
        self.chat_history = [summary_message] + remaining_messages
        
        print(f"Summarized {len(messages_to_summarize)} messages. "
              f"Extracted entities from conversation chunk.")
        
        # If still over limit, recursively summarize more
        if self.count_tokens(self.chat_history) > max_tokens:
            self.manage_token_limit(max_tokens, percent_to_summarize)

class SmartQueries(BaseModel):
    clarify: bool
    queryDB: bool
    queries: list[str]
    
class ChatResponse(BaseModel):
    response: str

class SummaryResponse(BaseModel):
    summary: str

def rewrite_query(user_query, client, currentSession: SessionState):
    # Get conversation history and entities (excluding current query since it hasn't been added yet)
    conversation_history = currentSession.format_conversation_history()
    extracted_entities = currentSession.format_extracted_entities()
    
    with open("prompts/rewrite_query.txt") as f:
        prompt_template = f.read()
    
    # Format the prompt with actual variables
    prompt = prompt_template.format(
        user_query=user_query,
        conversation_history=conversation_history,
        extracted_entities=extracted_entities
    )
    raw_response = client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.user_id,
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
    # Update conversation history with user query
    currentSession.update_chat_history("user", user_query)
    
    context = ""
    
    query_analysis = rewrite_query(user_query, client, currentSession)
    queries = query_analysis.queries

    if queries: 
        context += query_db(queries, top_k)
    

    # Use SessionState methods to format data for prompt
    conversation_history = currentSession.format_conversation_history()
    extracted_entities = currentSession.format_extracted_entities()

    with open("prompts/rag_bot.txt") as f:
        prompt_template = f.read()
    
    # Format the prompt with actual variables
    prompt = prompt_template.format(
        user_query=user_query,
        conversation_history=conversation_history,
        extracted_entities=extracted_entities,
        context=context,
        query_analysis=query_analysis
    )


    raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.user_id,
        text_format=ChatResponse)

    currentSession.last_response_id = raw_response.id
    parsed = raw_response.output_parsed
    
    # Update conversation history with assistant response
    currentSession.update_chat_history("assistant", parsed.response)
    
    # Output conversation state for visibility
    print(f"\n--- CONVERSATION STATE ---")
    print(f"Total messages in history: {len(currentSession.chat_history)}")
    print(f"Current token count: {currentSession.count_tokens(currentSession.chat_history)}")
    print(f"Extracted entities count: {len(currentSession.extracted_entities)}")
    if currentSession.extracted_entities:
        print("Recent entities:", [e['text'] for e in currentSession.extracted_entities[-5:]])
    print("--- END STATE ---\n")

    return parsed.response

def plan_discovery_node(user_query: str, currentSession: SessionState):
    """ This function systematically collects business size, location, and coverage preference information
    to help find eligible insurance plans. """

    print(f"\n=== PLAN DISCOVERY DEBUG ===")
    
    conversation_history = currentSession.format_conversation_history()
    current_answers = currentSession.plan_discovery_answers.model_dump_json() if currentSession.plan_discovery_answers else "{}"
    
    print(f"Current answers: {current_answers}")
    print(f"Chat history length: {len(currentSession.chat_history)} messages")
    print(f"Extracted entities count: {len(currentSession.extracted_entities)}")
    if currentSession.extracted_entities:
        print(f"Current entities: {[e['text'] for e in currentSession.extracted_entities]}")

    with open("prompts/plan_discovery.txt") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        user_query=user_query,
        conversation_history=conversation_history,
        current_answers=current_answers
    )
    
    print(f"Sending request to LLM...")

    raw_response = client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "developer", "content": prompt}],
        user=currentSession.user_id,
        text_format=PlanDiscoveryResponse)

    currentSession.last_response_id = raw_response.id
    parsed = raw_response.output_parsed
    
    print(f"LLM Response received!")
    print(f"Extracted answers: {parsed.plan_discovery_answers}")
    
    currentSession.plan_discovery_answers = parsed.plan_discovery_answers
    
    # Update chat history
    currentSession.update_chat_history("user", user_query)
    currentSession.update_chat_history("assistant", parsed.response)
    
    print(f"=== END DEBUG ===\n")

    return parsed.response



# Chat loop
if __name__ == "__main__":
    currentSession = SessionState()

    # Conversational Loop 
    print("Hello, welcome to Cigna Health Insurance! I am a helpful bot designed to help your company buy health insurance.")
    print("To begin, could you please tell me a bit about your business and its health insurance needs?")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        response = plan_discovery_node(query, currentSession)

        print("\nAssistant:", response)


