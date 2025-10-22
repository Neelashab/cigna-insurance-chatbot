import requests                                                                
import time                                                                   
import json                                                                   
                                                                       
# Configuration                                                                 
API_BASE_URL = "http://localhost:8000"                                    
                                                          
# Your links to upload                                                       
plan_links = [
    "https://www.cigna.com/individuals-families/shop-plans/plans-through-employer/open-access-plus", # OAP
    "https://www.cigna.com/employers/medical-plans/localplus?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # LP
    "https://www.cigna.com/employers/medical-plans/hmo?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # HMO
    "https://www.cigna.com/employers/medical-plans/network?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # MN
    "https://www.cigna.com/employers/medical-plans/ppo?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # PPO
    "https://www.cigna.com/employers/medical-plans/surefit?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # SF
    "https://www.cigna.com/employers/medical-plans/indemnity?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # MI
    "https://www.cigna.com/employers/small-business/small-group-health-insurance-plans?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # SG 
]

def upload_local_models(): 
    print("Uploading local insurance models to MongoDB (one time use)")
    try: 
        response = requests.post(f"{API_BASE_URL}/data/upload-local-models")
        if response.status_code == 200: 
            result = response.json()
            print("✅ Local models uploaded successfully!")
            print(f"  Models ID: {result.get('models_id', 'unknown')}")
            return True 
        else: 
            print(f"❌ Failed to upload models: {response.status_code}")
            return False 
    except Exception as e: 
        print(f"❌ Error uploading models: {e}")                                                                                         
        return False

def upload_scraped_documents(urls, job_name):
    print("\nSTARTING DOCUMENT UPLOAD JOB\n")
    try: 
        response = requests.post(f"{API_BASE_URL}/data/scrape-and-process", 
                                 json = {
                                     "urls": urls,
                                     "job_name": job_name})
        if response.status_code == 200:
            job_info = response.json()
            print(f"✅ Job started successfully!")
            print(f"  Job ID: {job_info['job_id']}")
            print(f"  Status: {job_info['status']}")
            print(f"  Message: {job_info['message']}")
            return job_info['job_id']
        else: 
            print(f"❌ Failed to start job: {response.status_code}")
            return None 
        
    except Exception as e: 
        print(f"Error starting job {e}")
        return None
    
def main():
    print("🚀 Starting Insurance Links Upload Script")
    print("=" * 50)

    upload_scraped_documents(plan_links, "upload cleaned documents")

        
if __name__ == "__main__":
    main()