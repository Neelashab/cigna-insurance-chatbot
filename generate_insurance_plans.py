import smart_scraper
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, create_model
from typing import Literal
import os
import json
import urllib.parse

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE = os.getenv("NAMESPACE")
HTML_CACHE_DIR = Path(os.getenv("HTML_CACHE_DIR"))
MONGODB_URI= os.getenv("MONGODB_URI")

# ---------------- MONGO SUPPORT ---------------------------

# URL encode the MongoDB URI to handle special characters
if MONGODB_URI and "://" in MONGODB_URI:
    # Parse the URI to extract and encode credentials
    from urllib.parse import quote_plus, urlparse
    parsed = urlparse(MONGODB_URI)
    if parsed.username and parsed.password:
        encoded_username = quote_plus(parsed.username)
        encoded_password = quote_plus(parsed.password)
        MONGODB_URI = MONGODB_URI.replace(f"{parsed.username}:{parsed.password}@", 
                                         f"{encoded_username}:{encoded_password}@")

# Create a new client and connect to the server
mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
db = mongo_client['cigna_insurance']
collection = db['insurance_plans']

# Send a ping to confirm a successful connection
try:
    mongo_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
# ----------------- END MONGO SUPPORT --------------------------


file_path = Path("insurance_models.py")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE = os.getenv("NAMESPACE")
HTML_CACHE_DIR = Path(os.getenv("HTML_CACHE_DIR"))

# SETUP 
client = OpenAI(api_key=OPENAI_API_KEY)
SESSION_ID = "generate_insurance_plans_id"  # For development, use a fixed session ID. In production, generate a new one each time.
plan_links = [
    "https://www.cigna.com/individuals-families/shop-plans/plans-through-employer/open-access-plus", # OAP
    "https://www.cigna.com/employers/medical-plans/localplus?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # LP
    "https://www.cigna.com/employers/medical-plans/hmo?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # HMO
    "https://www.cigna.com/employers/medical-plans/network?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # MN
    "https://www.cigna.com/employers/medical-plans/ppo?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # PPO
    "https://www.cigna.com/employers/medical-plans/surefit?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # SF
    "https://www.cigna.com/employers/medical-plans/indemnity?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # MI
    "https://www.cigna.com/employers/small-business/small-group-health-insurance-plans?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # SG 
    # ^ INCLUDE PDF FOR SG
]

state = Literal["All states",
        "AL", "AK", "AZ", "AR", "CA", "CO", 
        "CT", "DE", "FL", "GA", "HI", "ID", 
        "IL", "IN", "IA", "KS", "KY", "LA", 
        "ME", "MD", "MA", "MI", "MN", "MS", 
        "MO", "MT", "NE", "NV", "NH", "NJ", 
        "NM", "NY", "NC", "ND", "OH", "OK", 
        "OR", "PA", "RI", "SC", "SD", "TN", 
        "TX", "UT", "VT", "VA", "WA", "WV", 
        "WI", "WY", "DC"]

class PlanAnalysis(BaseModel):
    required_fields: list[str]
    key_differences: list[list[str]]

# HELPTER FUNCTIONS
def aggregate_page_contents(documents: list[Document], separator: str = "---NEW DOCUMENT---") -> str:
    """Aggregates the page_content of each LangChain Document into a single string with separators."""
    return separator.join(doc.page_content.strip() for doc in documents)

# Convert JSON to Pydantic models
def generate_pydantic_models(required_fields: list[str], key_differences: list[list[str]]):
    fields = {name: (str, ...) for name in required_fields}
    DynamicInsurancePlanModel = create_model("DynamicInsurancePlanModel", **fields)

    metadata = {}
    for tag in key_differences:
        tag_name = tag[0]
        literal_values = tuple(tag[1:])
        metadata[tag_name] = Literal[literal_values]

    metadata["location_availability"] = list[state]
    
    DynamicMetadataTags = create_model("DynamicMetaDataTags", **metadata)

    lines = [
        "from pydantic import BaseModel",
        "from typing import Literal",
        "",
        f"class {DynamicInsurancePlanModel}(BaseModel):"
    ]

    return DynamicInsurancePlanModel, DynamicMetadataTags

# Generate metadata for current document
def generate_metadata_tagger(Metadata: BaseModel):
    llm = ChatOpenAI(temperature=0, model="gpt-4.1")
    document_transformer = create_metadata_tagger(Metadata, llm)

# MAIN FUNCTIONS

# Generate fields for insurance models and write them to insurance_models.py
def plan_analysis(cleaned_plans: list[Document]) -> None:

    all_docs = aggregate_page_contents(cleaned_plans)

    with open("prompts/plan_analysis.txt") as f:
        prompt = f.read()

    raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        user=SESSION_ID,
        text_format=PlanAnalysis)

    parsed = raw_response.output_parsed

    model_input_data = {
    "required_fields": parsed.required_fields,
    "key_differences": parsed.key_differences
    }       

    # Write to file
    with open(file_path, "w") as f:
        json.dump(model_input_data, f, indent=2)

# Fit info into models
def fit_info_into_models(cleaned_plans: list[Document], InsuranceModel: BaseModel, Metadata: BaseModel):
    with open("prompts/insurance_model.txt") as f:
        insurance_model_prompt = f.read()

    with open("prompts/metadata.txt") as f:
        metadata_prompt = f.read()

    insurance_plans = []

    for idx, doc in enumerate(cleaned_plans, 1):
        IM_raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": f"{insurance_model_prompt}\n{doc.page_content}"}],
        user=SESSION_ID,
        text_format=InsuranceModel)

        IM_parsed = IM_raw_response.output_parsed
        page_content = IM_parsed.model_dump_json()

        # print(f"\n--- Parsed Insurance Plan #{idx} ---")
        # for field_name, value in IM_parsed.model_dump().items():
        #     print(f"{field_name}: {value}")
        # print("--- End of Plan ---\n")

        MD_raw_response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": f"{metadata_prompt}\n{doc.page_content}"}],
        user=SESSION_ID,
        text_format=Metadata)

        MD_parsed = MD_raw_response.output_parsed
        metadata = MD_parsed.model_dump()

        # print(f"\n--- Parsed Metadata #{idx} ---")
        # for field_name, value in MD_parsed.model_dump().items():
        #     print(f"{field_name}: {value}")

        print("METADATA TYPE: ", type(metadata))

        insurance_plans.append(Document(page_content=page_content, metadata=metadata))
        print("--- End of Plan ---\n")

    return insurance_plans

def generate_page_metadata(page_content: str, page_url: str) -> dict:
    """
    Generate metadata for a single scraped page using OpenAI reasoning model.
    Returns metadata as a dictionary.
    """
    print(f"Generating metadata for: {page_url}")
    
    with open("prompts/metadata.txt") as f:
        metadata_prompt = f.read()

    try:
        raw_response = client.responses.parse(
            model="o4-mini",
            input=[{"role": "user", "content": f"{metadata_prompt}\n\n{page_content}"}],
            user=SESSION_ID,
            reasoning={"effort": "medium"},
            text_format=DynamicMetaDataTags
        )
        
        metadata = raw_response.output_parsed.model_dump()
        print(f"  Generated metadata with {len(metadata)} fields")
        return metadata
        
    except Exception as e:
        print(f"  Error generating metadata: {e}")
        return {}

def upload_to_mongodb(page_content: str, metadata: dict, page_url: str):
    """
    Upload a document to MongoDB with metadata and raw text content.
    """
    try:
        # Create document with metadata fields + raw_text field
        document = metadata.copy()
        document['raw_text'] = page_content
        document['source_url'] = page_url
        
        # Insert into MongoDB collection
        result = collection.insert_one(document)
        print(f"  Uploaded to MongoDB with ID: {result.inserted_id}")
        return result.inserted_id
        
    except Exception as e:
        print(f"  Error uploading to MongoDB: {e}")
        return None

def process_pages_to_mongodb(cleaned_plans: list[Document]):
    """
    Process each scraped page individually and upload to MongoDB.
    """
    print(f"\n=== PROCESSING {len(cleaned_plans)} PAGES TO MONGODB ===")
    
    uploaded_count = 0
    
    for idx, doc in enumerate(cleaned_plans, 1):
        print(f"\n--- Processing Page {idx}/{len(cleaned_plans)} ---")
        page_url = doc.metadata.get('source', f'page_{idx}')
        
        # Generate metadata for this page
        metadata = generate_page_metadata(doc.page_content, page_url)
        
        if metadata:
            # Upload to MongoDB
            doc_id = upload_to_mongodb(doc.page_content, metadata, page_url)
            if doc_id:
                uploaded_count += 1
        
        print(f"--- End Page {idx} ---")
    
    print(f"\n=== MONGODB UPLOAD COMPLETE ===")
    print(f"Successfully uploaded {uploaded_count}/{len(cleaned_plans)} documents")
    return uploaded_count
        

if __name__ == "__main__":
    # scrape all plan info

    print("--- Starting Cigna Plan Processing Script ---")
    print("INSURANCE MODELS DO NOT EXIST, GENERATING THEM NOW...")
    print("\n=== PART 1: SCRAPING AND STORING DATA ===")
    retrieved_html_data = smart_scraper.scrape_and_store_if_not_exists(plan_links)

    print("\n=== PART 2: PARSE AND CLEAN HTML CONTENT ===")
    clean_data = smart_scraper.clean_data(retrieved_html_data)

    if not file_path.exists():
        print("\n=== PART 2.5: ANALYZE PLANS FOR REQUIRED FIELDS AND KEY DIFFERENCES ===")
        plan_analysis(clean_data)

    else: 
        print("USING EXISTING INSURANCE MODELS....")
    
    print("\n=== PART 3: LOAD MODEL INPUTS ===")
    with open("insurance_models.py") as f:
        data = json.load(f)

    DynamicInsurancePlanModel, DynamicMetaDataTags = generate_pydantic_models(
    required_fields=data["required_fields"],
    key_differences=data["key_differences"])

    print("\n\nMODEL FIELDS:\n\n")
    for model in [DynamicInsurancePlanModel, DynamicMetaDataTags]:
        for field_name, field_info in model.model_fields.items():
            print(f"{field_name}: {field_info.annotation}")
        print("\n\n")


    print("\n=== PART 4: PROCESS PAGES AND UPLOAD TO MONGODB ===")
    uploaded_count = process_pages_to_mongodb(clean_data)
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total pages processed: {len(clean_data)}")
    print(f"Successfully uploaded to MongoDB: {uploaded_count}")
    print(f"MongoDB Collection: cigna_insurance.insurance_plans")

    print("\n--- Cigna Data Processing Script Finished ---")





# You look through the plan links
    # ROUND 1 -> tell me what fields are in each plan? what do you feel is relevant to include? 
                # what are the key things that differentiate each plan and would influence an employeers decision when buying it? 

    # ROUND 2 -> go through the material again and fit it into the fields you identified in the first round. then upload these to pinecone

# create master list of all lists 
# upload each of the items in the list to pinecone 


# fully_insured_plan = """

#     # one for each state, size, industry
#     state: str
#     size: Literal["2–99", "100–499", "500–2,999", "3,000+"]
#     industry: Literal[
#         "Hospital and Health Systems", "Higher Education", "K-12 Education",
#         "State and Local Governments", "Taft-Hartley and Federal",
#         "Third Party Administrators (Payer Solutions)"
#     ]

#     plan_type: Literal["OAP", "PPO", "EPO", "LP", "SF", "HMO", "MN", "MI"]
#     pcp_required: bool 
#     pcp_auto_assign: bool 
#     refferal_to_specialist: bool
#     network_type: Literal["National", "Local"]
#     out_of_network_coverage: bool
#     urgent_care_coverage: bool
#     prior_authorization_required: bool
#     self_funded_options: bool
#     open_to_HSA: bool
#     coverage_highlights: str
#     plan_info: str
#     """

# Different Plan Types
# Set rules for how care is accessed 
# Affects cost structure but within a comparable range
"""
OAP -> Open Access Plan (all states and sizes)
PPO -> preferred provider organization (all states and sizes)
HMO -> health maintenance organization (all states and sizes)
EPO -> Exclusive Provider Information -> (all states, all sizes)
MN -> Medical Network (all states, all sizes)
MI -> Medical Indemnity (all states, all sizes)

## randomly generate eligble states
LP -> LocalPLus (some states, all sizes)
SF -> SureFit (some states, all sizes)

SG -> Small Group -> (TN, GA, AZ, 2-50 employees)
"""

# Cost Differences Between Plans 

# Tiers within each plan -> these dominate cost
"""
Tier -> Bronze, Silver, Gold, Platinum
Monthly Premium -> Low, Moderate, High. Very High 
OOP -> Out of Pocket Maximum -> High, Moderate, Low, Very Low
Copays -> None (pay full cost till deductible), Some, Most Services, Low Copay or None
Perks -> Few, Limited extras, Wellness/Telehealth, Most perks
"""

# Industry Specific Info
"""
Specific Group Benefits 
e.g Smart Support Program for specialized customer service for public sector clients
"""

