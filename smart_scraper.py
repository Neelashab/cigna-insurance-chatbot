import bs4
import requests
from pathlib import Path
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE = os.getenv("NAMESPACE")
HTML_CACHE_DIR = Path(os.getenv("HTML_CACHE_DIR"))

# SETUP 
client = OpenAI(api_key=OPENAI_API_KEY)
SESSION_ID = "development_id"  # For development, use a fixed session ID. In production, generate a new one each time.


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

TEST = ["https://www.cigna.com/individuals-families/shop-plans/plans-through-employer/open-access-plus"]

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

def parse_page(html: str) -> str:
    soup = bs4.BeautifulSoup(html, "html.parser")

    # manually determined. LLM could not do this reliably
    SLOT_TAGS_TO_DECOMPOSE = {
    "div": {"slot": ["disclaimer", "copyright", "primary-nav-search", "primary-nav-search-input-label", "primary-nav-language"]},
    "leaf-list": {"slot": ["legal-links", "main-links"]}}

    for tag_name, attrs in SLOT_TAGS_TO_DECOMPOSE.items():
        for slot_val in attrs.get("slot", []):
            for tag in soup.find_all(tag_name, attrs={"slot": slot_val}):
                tag.decompose()

    for li in soup.find_all("li"):
        li.decompose()

    for tag in soup.find_all("chc-skiplink"):
        tag.decompose()
    
    for tag in soup.find_all(attrs={"slot": "heading"}):
        tag.decompose() 

    extracted = soup.get_text(separator='\n', strip=True)

    return extracted

def batch_upsert_to_pinecone(index, records, namespace="ns2", batch_size=96):
    """Upserts records to Pinecone in batches."""
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        print(f"Upserting batch of {len(batch)} records to namespace '{namespace}'...")
        index.upsert_records(namespace, batch) 

def clean_data(data_to_process: list[dict]):
    """
    Cleans and chunks the provided data (list of dicts with 'url' and 'html')
    """
    if not data_to_process:
        print("No data provided to chunk and upload. Skipping.")
        return

    print("CLEANING DOCUMENT CONTENT...")
    cleaned_docs_for_langchain = []
    for item in data_to_process:
        text_content = parse_page(item["html"]) 
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
    
    return cleaned_docs_for_langchain

def chunk_data(cleaned_docs_for_langchain: list[Document]) -> list[Document]:

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
    
    return document_chunks

def upload_data(document_chunks: list[Document], pinecone_api_key: str, pinecone_index_host: str, namespace: str = "ns2"):
    
    print("SETTING UP PINECONE...")

    # create pinecone client
    try:
        pc = Pinecone(api_key=pinecone_api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize Pinecone client: {e}")
        return

    try:
        index = pc.Index(host=PINECONE_INDEX_HOST)
        print(f"Successfully connected to index")
    except Exception as e:
        print(f"ERROR: Failed to connect to Pinecone index: {e}")
        return

    # prepare records for upsert
    print("PREPARING RECORDS FOR PINECONE...")
    records_for_pinecone = []
    for i, chunk_doc in enumerate(document_chunks):
        pinecone_metadata = chunk_doc.metadata

        for key, value in pinecone_metadata.items():
            if not isinstance(value, (str, int, float, bool, list)):
                pinecone_metadata[key] = str(value) # Convert to string

        records_for_pinecone.append({
            "id": f"doc-{i}",
            "chunk_text": chunk_doc.page_content,
            **pinecone_metadata
        })

    if not records_for_pinecone:
        print("No records to upload after processing chunks.")
        return
        
    print(f"Prepared {len(records_for_pinecone)} records for upsertion.")
    
    print("UPLOADING TO PINECONE...")
    try:
        batch_upsert_to_pinecone(index, records_for_pinecone, namespace)
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
    retrieved_html_data = scrape_and_store_if_not_exists(TEST)

    print("\n=== PART 2: PARSE AND CLEAN HTML CONTENT ===")
    clean_data = clean_data(retrieved_html_data)
    print(f"Cleaned data: \n", clean_data)

    print("\n=== PART 3: SMARTLY CHUNK CLEANED CONTENT ===")
    document_chunks = chunk_data(clean_data)

    print("\n=== PART 4: UPLOAD TO DATABASE ===")
    #upload_data(document_chunks, PINECONE_API_KEY, PINECONE_INDEX_HOST)

    print("\n--- Cigna Data Processing Script Finished ---")


