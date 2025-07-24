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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE = os.getenv("NAMESPACE")
HTML_CACHE_DIR = Path(os.getenv("HTML_CACHE_DIR"))

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
        insurance_model_prompt = f.read()

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


    print("\n=== PART 4: FIT INSURANCE INFO INTO MODELS ===")
    insurance_plans = fit_info_into_models(clean_data, DynamicInsurancePlanModel, DynamicMetaDataTags)

    print("\n=== PART 5: CHUNK AND UPLOAD INSURANCE CONTENT ===")
    chunked_data = smart_scraper.chunk_data(insurance_plans)
    smart_scraper.upload_data(chunked_data, PINECONE_API_KEY, PINECONE_INDEX_HOST, "insurance_models")

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

