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
            print("✅ API Health Check:")
            print(f"  MongoDB: {health.get('mongodb', 'unknown')}")
            print(f"  Plans count: {health.get('mongodb_plans_count', 0)}")
            print(f"  Models in MongoDB: {health.get('models_in_mongodb', False)}")
            print(f"  Scraped docs count: {health.get('mongodb_scraped_count', 0)}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print("Make sure your server is running with: python api.py")
        return False

def upload_local_models():
    """Upload local insurance_models.py to MongoDB (one-time use)"""
    print("\n🔄 Uploading local Pydantic models to MongoDB...")
    try:
        response = requests.post(f"{API_BASE_URL}/data/upload-local-models")
        if response.status_code == 200:
            result = response.json()
            print("✅ Local models uploaded successfully!")
            print(f"  Models ID: {result.get('models_id', 'unknown')}")
            return True
        else:
            print(f"❌ Failed to upload models: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error uploading models: {e}")
        return False

def start_upload_job(urls, job_name):
    """Start the scrape and process job"""
    print(f"\n🚀 Starting upload job: {job_name}")
    print(f"📄 Uploading {len(urls)} URLs...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/data/scrape-and-process", 
            json={
                "urls": urls,
                "job_name": job_name
            }
        )
        
        if response.status_code == 200:
            job_info = response.json()
            print(f"✅ Job started successfully!")
            print(f"  Job ID: {job_info['job_id']}")
            print(f"  Status: {job_info['status']}")
            print(f"  Message: {job_info['message']}")
            return job_info['job_id']
        else:
            print(f"❌ Failed to start job: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error starting job: {e}")
        return None

def check_job_status(job_id):
    """Check the status of a job"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/jobs/{job_id}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error checking job status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking job status: {e}")
        return None

def wait_for_job_completion(job_id, max_wait_time=600):
    """Wait for job to complete, checking status every 10 seconds"""
    print(f"\n⏳ Waiting for job {job_id} to complete...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status_info = check_job_status(job_id)
        
        if not status_info:
            print("❌ Could not get job status")
            return False
            
        status = status_info['status']
        progress = status_info.get('progress', 'No progress info')
        
        print(f"  Status: {status} - {progress}")
        
        if status == 'completed':
            print("✅ Job completed successfully!")
            result = status_info.get('result', {})
            print("📊 Results:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            return True
        elif status == 'failed':
            print("❌ Job failed!")
            error = status_info.get('error', 'Unknown error')
            print(f"  Error: {error}")
            return False
        elif status in ['pending', 'running']:
            time.sleep(10)  # Wait 10 seconds before checking again
        else:
            print(f"⚠️  Unknown status: {status}")
            time.sleep(10)
    
    print("⏰ Job timed out - check manually")
    return False

def main():
    """Main function to run the upload process"""
    print("🚀 Starting Cigna Insurance Links Upload Script")
    print("=" * 50)
    
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
        print("\n🎉 All done! Your links have been processed and uploaded to MongoDB.")
        print("\n📋 What happened:")
        print("  1. URLs were scraped and cleaned")
        print("  2. Content stored in 'scraped_documents' collection")
        print("  3. Insurance models analyzed/loaded")
        print("  4. Final processed data stored in 'insurance_plans' collection")
        
        # Final health check
        print("\n📊 Final Status:")
        check_api_health()
    else:
        print("\n❌ Upload process failed. Check the logs above for details.")

if __name__ == "__main__":
    main()