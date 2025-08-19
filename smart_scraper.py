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
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import urllib.parse

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
NAMESPACE = os.getenv("NAMESPACE")
HTML_CACHE_DIR = Path(os.getenv("HTML_CACHE_DIR"))
MONGODB_URI = os.getenv("MONGODB_URI")

# MongoDB setup
if MONGODB_URI and "://" in MONGODB_URI:
    from urllib.parse import quote_plus, urlparse
    parsed = urlparse(MONGODB_URI)
    if parsed.username and parsed.password:
        encoded_username = quote_plus(parsed.username)
        encoded_password = quote_plus(parsed.password)
        MONGODB_URI = MONGODB_URI.replace(f"{parsed.username}:{parsed.password}@", 
                                         f"{encoded_username}:{encoded_password}@")

mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
db = mongo_client['cigna_insurance']
scraped_collection = db['scraped_documents']  # New collection for scraped documents

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

def check_url_exists_in_mongodb(url: str) -> bool:
    """Check if a URL already exists in MongoDB scraped documents collection"""
    try:
        existing = scraped_collection.find_one({"url": url})
        return existing is not None
    except Exception as e:
        print(f"Error checking URL in MongoDB: {e}")
        return False

def load_document_from_mongodb(url: str) -> Document:
    """Load a scraped document from MongoDB by URL"""
    try:
        doc = scraped_collection.find_one({"url": url})
        if not doc:
            raise Exception(f"Document not found for URL: {url}")
        
        return Document(
            page_content=doc["cleaned_content"],
            metadata={"source": doc["url"], "scraped_at": doc["scraped_at"]}
        )
    except Exception as e:
        print(f"Error loading document from MongoDB: {e}")
        raise

def save_cleaned_document_to_mongodb(url: str, cleaned_content: str):
    """Save cleaned document content to MongoDB"""
    try:
        document = {
            "url": url,
            "cleaned_content": cleaned_content,
            "scraped_at": datetime.now()
        }
        
        # Use upsert to replace existing document if URL already exists
        result = scraped_collection.update_one(
            {"url": url},
            {"$set": document},
            upsert=True
        )
        
        if result.upserted_id:
            print(f"STORED: Saved cleaned content for {url} to MongoDB with ID: {result.upserted_id}")
        else:
            print(f"UPDATED: Updated cleaned content for {url} in MongoDB")
            
        return result.upserted_id or result.matched_count
        
    except Exception as e:
        print(f"Error saving document to MongoDB: {e}")
        raise

def scrape_and_store_if_not_exists(urls_to_scrape: list[str]) -> list[Document]:
    """
    Scrapes and cleans HTML content from URLs. If a URL's content is already in MongoDB,
    it loads from there. Otherwise, it fetches, cleans, stores in MongoDB, and returns cleaned Documents.
    """
    processed_documents = []

    for url in urls_to_scrape:
        try:
            # Check if URL already exists in MongoDB
            if check_url_exists_in_mongodb(url):
                print(f"MONGO HIT: Loading cleaned content for {url} from MongoDB")
                document = load_document_from_mongodb(url)
                processed_documents.append(document)
            else:
                print(f"MONGO MISS: Fetching and cleaning content for {url}")
                
                # Fetch HTML content
                res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                res.raise_for_status()
                html_content = res.text
                
                # Clean the content immediately
                cleaned_content = parse_page(html_content)
                
                if cleaned_content:
                    # Save cleaned content to MongoDB
                    save_cleaned_document_to_mongodb(url, cleaned_content)
                    
                    # Create Document object for return
                    document = Document(
                        page_content=cleaned_content,
                        metadata={"source": url, "scraped_at": datetime.now()}
                    )
                    processed_documents.append(document)
                else:
                    print(f"WARNING: No content after cleaning for {url}")
                    
        except requests.exceptions.RequestException as e:
            print(f"FETCH FAILED for {url}: {e}")
        except Exception as e:
            print(f"PROCESSING ERROR for {url}: {e}")
                
    print(f"Finished scraping/loading. Total cleaned documents: {len(processed_documents)}")
    return processed_documents

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

def clean_data(scraped_documents: list[Document]) -> list[Document]:
    """
    DEPRECATED: This function is now a pass-through since cleaning happens during scraping.
    Kept for backward compatibility - just returns the already-cleaned documents.
    """
    print("Note: Documents are already cleaned during scraping. Returning as-is.")
    
    if scraped_documents:
        print(f"📄 Example cleaned content (first document, first 300 chars):\n{scraped_documents[0].page_content[:300]}...\n")
    
    return scraped_documents

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
    print("--- Starting Smart Scraper Test Script ---")

    print("\n=== PART 1: SCRAPING AND CLEANING DATA (MongoDB) ===")
    cleaned_documents = scrape_and_store_if_not_exists(TEST)

    print("\n=== PART 2: CHUNKING CLEANED CONTENT ===")
    document_chunks = chunk_data(cleaned_documents)

    print("\n=== PART 3: UPLOAD TO PINECONE (Optional) ===")
    #upload_data(document_chunks, PINECONE_API_KEY, PINECONE_INDEX_HOST)

    print("\n--- Smart Scraper Test Finished ---")


