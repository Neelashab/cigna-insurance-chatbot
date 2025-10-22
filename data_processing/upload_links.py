import requests
import time
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

# Your links to upload
links = [
    "https://www.cigna.com/employers/medical-plans/ppo",
    "https://www.cigna.com/employers/medical-plans/hmo",
    "https://www.cigna.com/employers/medical-plans/localplus",
    "https://www.cigna.com/employers/medical-plans/open-access-plus",
    "https://www.cigna.com/employers/medical-plans/surefit",
    "https://www.cigna.com/employers/medical-plans/indemnity",
    "https://www.cigna.com/employers/small-business/small-group-health-insurance-plans",
    # Add more links here as needed
]

def check_api_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/health")
        if response.status_code == 200:
            health = response.json()
            return True
        else:
            return False
    except Exception as e:
        return False

def upload_local_models():
    """Upload local insurance_models.py to MongoDB (one-time use)"""
    try:
        response = requests.post(f"{API_BASE_URL}/data/upload-local-models")
        if response.status_code == 200:
            result = response.json()
            return True
        else:
            return False
    except Exception as e:
        return False

def start_upload_job(urls, job_name):
    """Start the scrape and process job"""
    
    try:
        response = requests.post(f"{API_BASE_URL}/data/scrape-and-process", 
            json={
                "urls": urls,
                "job_name": job_name
            }
        )
        
        if response.status_code == 200:
            job_info = response.json()
            return job_info['job_id']
        else:
            return None
    except Exception as e:
        return None

def check_job_status(job_id):
    """Check the status of a job"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/jobs/{job_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

def wait_for_job_completion(job_id, max_wait_time=600):
    """Wait for job to complete, checking status every 10 seconds"""
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status_info = check_job_status(job_id)
        
        if not status_info:
            return False
            
        status = status_info['status']
        progress = status_info.get('progress', 'No progress info')
        
        
        if status == 'completed':
            result = status_info.get('result', {})
            return True
        elif status == 'failed':
            return False
        elif status in ['pending', 'running']:
            time.sleep(10)  # Wait 10 seconds before checking again
        else:
            time.sleep(10)
    
    return False

def main():
    """Main function to run the upload process"""
    
    # Step 1: Check API health
    if not check_api_health():
        return
    
    # Step 2: Upload local models (one-time)
    upload_local_models()
    
    # Step 3: Start upload job
    job_id = start_upload_job(links, "bulk_cigna_links_upload")
    
    if not job_id:
        return
    
    # Step 4: Wait for completion
    success = wait_for_job_completion(job_id)
    
    if success:
        # Final health check
        check_api_health()
    else:

if __name__ == "__main__":
    main()