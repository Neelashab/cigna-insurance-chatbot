
import bs4
import random
import requests
import json
from pathlib import Path
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
HTML_CACHE_DIR = os.getenv("HTML_CACHE_DIR")

# SETUP 
client = OpenAI(api_key=OPENAI_API_KEY)
SESSION_ID = "development_id"  # For development, use a fixed session ID. In production, generate a new one each time.


class HTMLTags(BaseModel):
    tags_to_parse: list[str]
    blocked_classes: list[str]
    blocked_headings: list[str]
    include_headers: bool 


# SCRAPING NOTES:
# - LLM does not have the content of the forms, should be instructed to tell customer a certain form is available on their website and what the title is
# - Medical and dental plan documents are here: https://www.cigna.com/individuals-families/member-guide/plan-documents/
#   * there are too many permutations for state and year to scrape all of them, instruct LLM accordingly

URLS = ["https://www.cigna.com/individuals-families/shop-plans/health-insurance-plans/",
                "https://www.cigna.com/individuals-families/member-guide/",
               "https://www.cigna.com/individuals-families/member-guide/mycigna",
               "https://www.cigna.com/individuals-families/member-guide/home-delivery-pharmacy",
               "https://www.cigna.com/individuals-families/member-guide/specialty-pharmacy",
               "https://www.cigna.com/individuals-families/member-guide/employee-assistance-program",
               "https://www.cigna.com/individuals-families/member-guide/customer-forms/",
               "https://www.cigna.com/individuals-families/shop-plans/health-insurance-plans/plan-benefits/#affordability-tag",
               "https://www.cigna.com/individuals-families/shop-plans/health-insurance-plans/plan-benefits/#24-7-virtual-care-tag",
               "https://www.cigna.com/individuals-families/shop-plans/health-insurance-plans/plan-benefits/#guided-customer-care-tag",
               "https://www.cigna.com/individuals-families/shop-plans/health-insurance-plans/plan-benefits/#personalized-digital-tools-tag",
               "https://www.cigna.com/individuals-families/shop-plans/health-insurance-plans/plan-benefits/#rewards-and-discounts-tag",
               "https://www.cigna.com/individuals-families/shop-plans/health-insurance-plans/plan-benefits/#customer-service-tag",
               "https://www.cigna.com/individuals-families/shop-plans/dental-insurance-plans/cigna-dental-1500",
               "https://www.cigna.com/individuals-families/shop-plans/supplemental/hospital-indemnity-insurance",
               "https://www.cigna.com/individuals-families/shop-plans/supplemental/lump-sum-cancer-insurance",
               "https://www.cigna.com/individuals-families/shop-plans/supplemental/cancer-treatment-insurance",
               "https://www.cigna.com/medicare/shop-plans/supplemental/plan-g?campaign_ID=CSBORG",
               "https://www.cigna.com/knowledge-center/choosing-a-medicare-plan",
               "https://www.cigna.com/medicare/shop-plans/medicare-advantage/special-needs-plans",
               "https://www.cigna.com/medicare/shop-plans/medicare-advantage/",
               "https://www.cigna.com/medicare/shop-plans/medicare-advantage/additional-benefits",
               "https://www.cigna.com/medicare/shop-plans/medicare-advantage/additional-benefits#part-b-giveback",
               "https://www.cigna.com/medicare/shop-plans/supplemental/?campaign_ID=CSBORG",
               "https://www.cigna.com/knowledge-center/what-is-medicare",
               "https://www.cigna.com/knowledge-center/what-is-medicare-part-a-part-b",
               "https://www.cigna.com/knowledge-center/what-is-medicare-part-d",
               "https://www.cigna.com/knowledge-center/what-is-medicare-supplement-insurance-medigap",
               "https://www.cigna.com/knowledge-center/part-b-part-d-coverage-differences",
               "https://www.cigna.com/knowledge-center/in-network-vs-out-of-network",
               "https://www.cigna.com/knowledge-center/open-enrollment-special-enrollment",
               "https://www.cigna.com/knowledge-center/health-insurance-marketplace",
               "https://www.cigna.com/knowledge-center/bronze-silver-gold-platinum-health-plans",
               "https://www.cigna.com/individuals-families/shop-plans/dental-insurance-plans/",
               "https://www.cigna.com/knowledge-center/full-coverage-dental-insurance",
               "https://www.cigna.com/knowledge-center/dental-hmo-vs-ppo-plans",
               "https://www.cigna.com/knowledge-center/orthodontic-insurance",
               "https://www.cigna.com/knowledge-center/dental-insurance-cost",
               "https://www.cigna.com/individuals-families/shop-plans/supplemental/"]


# Scrape and store HTMl if it doesn't exist in local cache
def url_to_filename(url: str) -> str:
    """Converts a URL to a safe filename."""
    name = re.sub(r'^https?://', '', url)
    name = re.sub(r'[<>:"/\\|?*&]', '_', name)
    name = name.strip('_.')
    # Limit length to avoid issues with very long URLs, though less critical for local cache
    max_len = 100 
    return name[:max_len] + ".html"

def scrape_and_store_if_not_exists(urls_to_scrape: list[str]) -> list[dict]:
    """
    Scrapes HTML content from URLs. If a URL's content is already cached locally,
    it loads from the cache. Otherwise, it fetches, stores it, and then returns it.
    """
    HTML_CACHE_DIR.mkdir(exist_ok=True)
    processed_data = []

    for url in urls_to_scrape:
        filename = url_to_filename(url)
        filepath = HTML_CACHE_DIR / filename
        
        html_content = None
        loaded_from_cache = False

        if filepath.exists():
            print(f"CACHE HIT: Attempting to load content for {url} from {filepath}")
            try:
                html_content = filepath.read_text(encoding='utf-8')
                processed_data.append({"url": url, "html": html_content})
                loaded_from_cache = True
            except Exception as e:
                print(f"CACHE READ ERROR for {filepath}: {e}. Will attempt to re-fetch.")
        
        if not loaded_from_cache:
            print(f"CACHE MISS or read error: Fetching content for {url}")
            try:
                res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                res.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                html_content = res.text
                
                filepath.write_text(html_content, encoding='utf-8')
                print(f"STORED: Saved content for {url} to {filepath}")
                processed_data.append({"url": url, "html": html_content})
            except requests.exceptions.RequestException as e:
                print(f"FETCH FAILED for {url}: {e}")
            except Exception as e: 
                print(f"An unexpected error occurred while fetching/storing {url}: {e}")
                
    print(f"Finished scraping/loading. Total documents for processing: {len(processed_data)}")
    return processed_data



# ------- LLM IN THE LOOP ---------- # 
# introduce LLM in the loop to decide on what to clean 

def sample_html_files(sample_size=3):
    """Returns a random sample of HTML files from the cache directory."""
    files = list(HTML_CACHE_DIR.glob("*.html"))
    return random.sample(files, min(sample_size, len(files)))


def smart_parse():
    """Give LLM a chance to parse HTML files and output tags to either ignore or parse"""

    paths = sample_html_files()
    raw_html = [path.read_text() for path in paths]
    clipped_html = "\n\n".join(html[:3000] for html in raw_html)

    prompt = f"""
You are a helpful assistant improving a web scraping pipeline for a health insurance website.

Your job is to analyze the raw HTML below and output a set of scraping rules.

Instructions:
- Identify and list the tags (including custom tags like <leaf-card>) that contain meaningful content.
- Identify any tags/classes/headings that should be ignored (like footers, navs, disclaimers, careers, etc.).
- Indicate whether structural headers (like h2, h3) should be captured to preserve context.
- If in doubt about a tag, include it in the output.

Output MUST be valid JSON using this structure:
{{
  "tags_to_parse": ["div", "leaf-card", "leaf-list"],
  "blocked_classes": ["legal-links", "footer", "nav-bar"],
  "blocked_headings": ["privacy", "careers", "sitemap"],
  "include_headers": true
}}

HTML TO ANALYZE:{clipped_html}"""

    response = client.responses.parse(
        model="gpt-4.1",
        input=[{"role": "developer", "content": prompt}],
        text_format=HTMLTags,
        user=SESSION_ID)

    return response.output_parsed


# ------- LLM IN THE LOOP ---------- # 




def clean_doc_content(html: str, tags:HTMLTags) -> str:
    """Parse raw HTML and clean content by disposing of blocked headers and slots"""
    soup = bs4.BeautifulSoup(html, "html.parser")
    extracted = []

    for tag in soup.find_all(tags.tags_to_parse):
        if tag.name == "div" and tags.blocked_classes:
            if any(cls in tag.get("class", []) for cls in tags.blocked_classes):
                continue
        
        heading_text = ""
        if tags.include_headers:
            for header_tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                heading = tag.find(header_tag)
                if heading:
                    heading_text = heading.get_text(strip=True).lower()
                    break

        if any(block in heading_text for block in tags.blocked_headings):
            continue

        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 30:
            extracted.append(text)

    return "\n\n".join(extracted)


def batch_upsert_to_pinecone(index, records, namespace="ns2", batch_size=96):
    """Upserts records to Pinecone in batches."""
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        print(f"Upserting batch of {len(batch)} records to namespace '{namespace}'...")
        index.upsert_records(namespace, batch) 

def chunk_and_upload_data(data_to_process: list[dict], pinecone_api_key: str, pinecone_index_name: str, tags: HTMLTags):
    """
    Cleans, chunks, and uploads the provided data (list of dicts with 'url' and 'html')
    to the specified Pinecone index.
    """
    if not data_to_process:
        print("No data provided to chunk and upload. Skipping.")
        return

    print("CLEANING DOCUMENT CONTENT...")
    cleaned_docs_for_langchain = []
    for item in data_to_process:
        text_content = clean_doc_content(item["html"], tags) 
        if text_content:
            cleaned_docs_for_langchain.append(
                Document(
                    page_content=text_content,
                    metadata={"source": item["url"]}
                )
            )
    
    if not cleaned_docs_for_langchain:
        print("No content remained after cleaning. Skipping further processing.")
        return
        
    print(f"Cleaned {len(cleaned_docs_for_langchain)} documents.")
    if cleaned_docs_for_langchain:
        print(f"📄 Example cleaned content (first document, first 300 chars):\n{cleaned_docs_for_langchain[0].page_content[:300]}...\n")

    print("SPLITTING INTO CHUNKS...")

    # TODO look into all text splitting strategies
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100, add_start_index=True
    )
    document_chunks = text_splitter.split_documents(cleaned_docs_for_langchain)
    print(f"Chunked into {len(document_chunks)} total splits.")

    if not document_chunks:
        print("No chunks created. Skipping upload to Pinecone.")
        return

    print("SETTING UP PINECONE...")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize Pinecone client: {e}")
        return

    # Pinecone index handling
    # TODO look into using pc.create_index instead?
    try:
        existing_indexes = pc.list_indexes().names()
    except Exception as e:
        print(f"ERROR: Failed to list Pinecone indexes: {e}. Please check API key and connection.")
        return

    if pinecone_index_name not in existing_indexes:
        print(f"Index '{pinecone_index_name}' not found. Attempting to create it using provided configuration...")
        try:
            pc.create_index_for_model(
                name=pinecone_index_name,
                cloud="aws",
                region="us-east-1", 
                embed={ # specify model to use for embedding & tell it to embed chunk text field
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"}
                }
            )
            print(f"Index '{pinecone_index_name}' creation request sent. It may take some time to become ready.")
            # TODO consider adding polling mechanism to wait for index to be ready
        except Exception as e:
            print(f"ERROR: Failed to create Pinecone index '{pinecone_index_name}': {e}")
            print("Please ensure the index is created manually or check the creation parameters and your Pinecone client version/setup.")
            return 
    else:
        print(f"Index '{pinecone_index_name}' already exists.")

    try:
        index = pc.Index(pinecone_index_name)
        print(f"Successfully connected to index '{pinecone_index_name}'.")
    except Exception as e:
        print(f"ERROR: Failed to connect to Pinecone index '{pinecone_index_name}': {e}")
        return

    print("PREPARING RECORDS FOR PINECONE...")
    records_for_pinecone = []
    for i, chunk_doc in enumerate(document_chunks):
        records_for_pinecone.append({
            "_id": f"doc-{i}",
            "chunk_text": chunk_doc.page_content, # field to embed
            "source": chunk_doc.metadata.get("source", "unknown"),
        })

    if not records_for_pinecone:
        print("No records to upload after processing chunks.")
        return
        
    print(f"Prepared {len(records_for_pinecone)} records for upsertion.")
    
    print("UPLOADING TO PINECONE...")
    try:
        batch_upsert_to_pinecone(index, records_for_pinecone, namespace="ns2")
    except Exception as e:
        print(f"ERROR: Failed during batch upsert to Pinecone: {e}")
        print("Please check the `batch_upsert_to_pinecone` function and the `index.upsert_records` call's compatibility with your Pinecone setup.")
        return

    try:
        stats = index.describe_index_stats()
        print("Pinecone index stats after upload attempt:")
        print(stats)
    except Exception as e:
        print(f"Could not retrieve index stats: {e}")



if __name__ == "__main__":
    print("--- Starting Cigna Data Processing Script ---")

    print("\n=== PART 1: SCRAPING AND STORING DATA ===")
    retrieved_html_data = scrape_and_store_if_not_exists(URLS)

    if not retrieved_html_data:
        print("No data was retrieved or loaded. Exiting script.")
        
    else:
        print("\n=== PART 2: SMART PARSING HTML CONTENT ===")
        tags = smart_parse()
        print("SMART PARSE OUTPUT: ", tags)
        print("\n=== PART 3: CHUNKING AND UPLOADING TO DATABASE ===")
        chunk_and_upload_data(retrieved_html_data, PINECONE_API_KEY, PINECONE_INDEX_NAME, tags)

    print("\n--- Cigna Data Processing Script Finished ---")

