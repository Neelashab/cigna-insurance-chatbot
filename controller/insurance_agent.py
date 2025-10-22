from pinecone import Pinecone
from openai import OpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import urllib.parse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import date
from typing import Literal 
import os
import tiktoken
import uuid
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from models.schemas import BusinessProfile, PlanDiscoveryResponse, PlanDiscoveryAnswers, SmartQueries, ChatResponse, SummaryResponse

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE = os.getenv("NAMESPACE")
MONGODB_URI = os.getenv("MONGODB_URI")

# Create client 
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)
client = OpenAI(api_key=OPENAI_API_KEY)


mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
db = mongo_client['cigna_insurance']
collection = db['insurance_plans']

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
    
    # Update conversation history with user query first
    currentSession.update_chat_history("user", user_query)
    
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
    
    # Update chat history with assistant response
    currentSession.update_chat_history("assistant", parsed.response)
    
    
    print(f"=== END DEBUG ===\n")

    return parsed.response

def map_business_size_to_categories(employee_count: int) -> list[str]:
    """
    Map employee count to all applicable business size categories.
    Returns list of categories that the business size qualifies for.
    """
    categories = []
    
    if 2 <= employee_count <= 50:
        categories.append("2-50")
    if 2 <= employee_count <= 99:
        categories.append("2-99")
    if 51 <= employee_count <= 99:
        categories.append("51-99")
    if 100 <= employee_count <= 499:
        categories.append("100-499")
    if 500 <= employee_count <= 2999:
        categories.append("500-2,999")
    if employee_count >= 3000:
        categories.append("3,000+")
    
    # Also add "All sizes" as it matches any business
    categories.append("All sizes")
    
    return categories

def search_eligible_plans(plan_answers: PlanDiscoveryAnswers):
    """
    Search MongoDB for insurance plans that match user's business profile.
    Returns a dictionary where key is plan name and value is summary text.
    """
    print(f"\n=== PLAN SEARCH DEBUG ===")
    print(f"Searching MongoDB for plans with:")
    print(f"  Business Size: {plan_answers.business_size}")
    print(f"  Location: {plan_answers.location}")
    print(f"  Coverage Preference: {plan_answers.coverage_preference}")
    
    # Build MongoDB query filters
    query_filters = {}
    
    # Coverage preference filter (Network Type)
    if plan_answers.coverage_preference:
        query_filters["Network Type"] = plan_answers.coverage_preference
    
    # Build list of filters to combine
    filters_list = []
    
    # Business size filter
    if plan_answers.business_size:
        size_categories = map_business_size_to_categories(plan_answers.business_size)
        filters_list.append({
            "$or": [
                {"Business Size Eligibility": {"$in": size_categories}},
                {"Business Size Eligibility": "All sizes"}
            ]
        })
    
    # Location filter
    if plan_answers.location:
        filters_list.append({
            "$or": [
                {"location_availability": {"$in": [plan_answers.location]}},
                {"location_availability": {"$in": ["All states"]}}
            ]
        })
    
    # Combine all filters
    if len(filters_list) > 1:
        query_filters["$and"] = filters_list
    elif len(filters_list) == 1:
        query_filters.update(filters_list[0])
    
    print(f"  MongoDB query: {query_filters}")
    
    # Query MongoDB
    try:
        cursor = collection.find(query_filters)
        matching_docs = list(cursor)
        print(f"  Found {len(matching_docs)} matching documents")
        
        # Create dictionary: plan_name -> summary
        plan_dict = {}
        for doc in matching_docs:
            plan_name = doc.get("Plan Type", "Unknown Plan")
            summary = doc.get("summary", "")
            
            if plan_name != "Unknown Plan" and summary:
                plan_dict[plan_name] = summary
                print(f"    Added plan: {plan_name}")
            else:
                print(f"    Skipped document with missing Plan Type or summary")
        
        print(f"  Returning {len(plan_dict)} unique plans")
        print(f"  Plan names: {list(plan_dict.keys())}")
        print(f"=== END PLAN SEARCH ===\n")
        
        return plan_dict
        
    except Exception as e:
        print(f"Error searching MongoDB: {e}")
        return {}

def reason_about_plans(eligible_plans: dict, plan_answers: PlanDiscoveryAnswers) -> str:
    """
    Use reasoning model to analyze and rank insurance plans based on business profile.
    Returns comprehensive analysis and recommendation.
    """
    print(f"\n=== REASONING ABOUT PLANS ===")
    print(f"Analyzing {len(eligible_plans)} eligible plans for business profile...")
    
    # Use stored summaries directly
    print("Using stored plan summaries...")
    plan_summaries = eligible_plans  # eligible_plans now contains summaries, not raw text
    
    # Create formatted summaries text
    summaries_text = "\n\n".join([f"=== {plan} ===\n{summary}" for plan, summary in plan_summaries.items()])
    
    with open("prompts/reason_about_plans.txt") as f:
        prompt_template = f.read()
    
    # Format the prompt with actual variables
    prompt = prompt_template.format(
        business_size=plan_answers.business_size,
        location=plan_answers.location,
        coverage_preference=plan_answers.coverage_preference,
        plan_summaries=summaries_text
    )

    try:
        response = client.responses.parse(
            model="o4-mini",
            input=[{"role": "user", "content": prompt}],
            user=str(uuid.uuid4()),
            reasoning={"effort": "medium"},
            text_format=ChatResponse
        )
        
        analysis_result = response.output_parsed.response
        
        print(f"\nPlan analysis and ranking complete!")
        print("="*60)
        print(analysis_result)
        print("="*60)
        
        return analysis_result
        
    except Exception as e:
        print(f"Error analyzing plans: {e}")
        return f"Error occurred during plan analysis. Available plans: {list(eligible_plans.keys())}"

def complete_insurance_workflow(currentSession: SessionState):
    """
    Orchestrates the complete insurance recommendation workflow:
    1. Plan discovery (collect business profile)
    2. Search eligible plans
    3. Reason about and rank plans
    """
    print(f"\n{'='*60}")
    print("STARTING COMPLETE INSURANCE WORKFLOW")
    print(f"{'='*60}")
    
    # Step 1: Plan Discovery
    print("\n=== STEP 1: PLAN DISCOVERY ===")
    print("Let's start by understanding your business needs...")
    
    # Run plan discovery loop until we have complete information
    while not (currentSession.plan_discovery_answers and 
               currentSession.plan_discovery_answers.business_size and
               currentSession.plan_discovery_answers.location and 
               currentSession.plan_discovery_answers.coverage_preference):
        
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            return "Workflow cancelled by user."
            
        response = plan_discovery_node(user_query, currentSession)
        print(f"\nAssistant: {response}")
    
    print(f"\n✓ Plan discovery complete!")
    print(f"  Business Size: {currentSession.plan_discovery_answers.business_size} employees")
    print(f"  Location: {currentSession.plan_discovery_answers.location}")
    print(f"  Coverage Preference: {currentSession.plan_discovery_answers.coverage_preference}")
    
    # Step 2: Search Eligible Plans
    print(f"\n=== STEP 2: SEARCHING ELIGIBLE PLANS ===")
    eligible_plans = search_eligible_plans(currentSession.plan_discovery_answers)
    
    if not eligible_plans:
        return "No eligible plans found for your business profile. Please contact us directly for assistance."
    
    print(f"✓ Found {len(eligible_plans)} eligible plans")
    
    # Step 3: Reason About Plans
    print(f"\n=== STEP 3: ANALYZING AND RANKING PLANS ===")
    analysis_result = reason_about_plans(eligible_plans, currentSession.plan_discovery_answers)
    
    print(f"\n✓ Analysis complete! Here's your personalized recommendation:")
    print(f"\n{analysis_result}")
    
    print(f"\n{'='*60}")
    print("INSURANCE WORKFLOW COMPLETE")
    print(f"{'='*60}")
    
    return analysis_result



# Chat loop
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("TESTING COMPLETE WORKFLOW")
    print("="*60)
    
    # Start the complete workflow
    currentSession = SessionState()
    print("Hello! I'm here to help you find the best health insurance plan for your business.")
    print("To get started, could you tell me about your business and its insurance needs?")
    
    complete_insurance_workflow(currentSession)


