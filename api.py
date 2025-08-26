from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
from enum import Enum
import uvicorn
import uuid
from datetime import datetime
import json
from pathlib import Path

# Import existing modules
from insurance_agent import (
    SessionState, 
    ask_rag_bot, 
    plan_discovery_node,
    search_eligible_plans,
    reason_about_plans
)
from schemas import PlanDiscoveryAnswers

# Import data processing modules
import smart_scraper
from generate_insurance_plans import (
    plan_analysis, 
    process_pages_to_mongodb, 
    generate_pydantic_models,
    mongo_client,
    collection,
    models_collection,
    plan_links,
    check_models_exist_in_mongodb,
    load_models_from_mongodb,
    upload_local_models_to_mongodb
)

app = FastAPI(
    title="Cigna Insurance Chatbot API",
    description="FastAPI service for Cigna insurance chatbot functionality and data processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (in production, use Redis or database)
sessions = {}

# Job status tracking for data processing
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

class JobInfo(BaseModel):
    job_id: str
    job_name: Optional[str] = None
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[str] = None

# Global job store (in production, use Redis or database)
jobs_store: Dict[str, JobInfo] = {}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

class PlanDiscoveryRequest(BaseModel):
    message: str

class PlanDiscoveryResponseModel(BaseModel):
    response: str
    session_id: str
    plan_discovery_answers: Optional[PlanDiscoveryAnswers] = None
    is_complete: bool = False

class PlanAnalysisResponse(BaseModel):
    analysis: str
    eligible_plans_count: int
    session_id: str

# Data processing models
class ScrapeRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None  # If None, use default plan_links
    job_name: Optional[str] = None

class ProcessRequest(BaseModel):
    job_name: Optional[str] = None

class ScrapeAndProcessRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None  # If None, use default plan_links
    job_name: Optional[str] = None

def create_session_id() -> str:
    """Create a new session ID"""
    return str(uuid.uuid4())

def get_session(session_id: str) -> SessionState:
    """Get existing session or raise error"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

# Job utility functions
def create_job(job_name: Optional[str] = None) -> str:
    """Create a new job and return job ID"""
    job_id = str(uuid.uuid4())
    job_info = JobInfo(
        job_id=job_id,
        job_name=job_name,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    jobs_store[job_id] = job_info
    return job_id

def update_job_status(job_id: str, status: JobStatus, result: Optional[Dict] = None, 
                     error: Optional[str] = None, progress: Optional[str] = None):
    """Update job status"""
    if job_id in jobs_store:
        job_info = jobs_store[job_id]
        job_info.status = status
        job_info.updated_at = datetime.now()
        if result:
            job_info.result = result
        if error:
            job_info.error = error
        if progress:
            job_info.progress = progress

def get_job_info(job_id: str) -> Optional[JobInfo]:
    """Get job information"""
    return jobs_store.get(job_id)

@app.get("/")
async def root():
    return {"message": "Cigna Insurance Chatbot API is running"}

@app.get("/test/hello")
async def test_hello():
    return {"message": "Hello from continuous deployment!", "timestamp": datetime.now().isoformat()}

@app.post("/session")
async def create_session():
    """Create a new session"""
    session_id = create_session_id()
    sessions[session_id] = SessionState()
    
    return {
        "session_id": session_id,
        "message": "Session created successfully"
    }

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_endpoint(session_id: str, request: ChatRequest):
    """General chat endpoint using RAG"""
    try:
        session = get_session(session_id)
        response = ask_rag_bot(request.message, session)
        
        return ChatResponse(
            response=response,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/plan-discovery/{session_id}", response_model=PlanDiscoveryResponseModel)
async def plan_discovery_endpoint(session_id: str, request: PlanDiscoveryRequest):
    """Plan discovery endpoint to collect business profile information"""
    try:
        session = get_session(session_id)
        response = plan_discovery_node(request.message, session)
        
        # Check if plan discovery is complete
        is_complete = (
            session.plan_discovery_answers is not None and
            session.plan_discovery_answers.business_size is not None and
            session.plan_discovery_answers.location is not None and
            session.plan_discovery_answers.coverage_preference is not None
        )
        
        return PlanDiscoveryResponseModel(
            response=response,
            session_id=session_id,
            plan_discovery_answers=session.plan_discovery_answers,
            is_complete=is_complete
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan discovery error: {str(e)}")

@app.post("/analyze-plans/{session_id}", response_model=PlanAnalysisResponse)
async def analyze_plans_endpoint(session_id: str):
    """Analyze and rank eligible plans based on collected business profile"""
    try:
        session = get_session(session_id)
        
        if not session.plan_discovery_answers:
            raise HTTPException(status_code=400, detail="Plan discovery not completed")
        
        # Check if all required fields are present
        if not all([
            session.plan_discovery_answers.business_size,
            session.plan_discovery_answers.location,
            session.plan_discovery_answers.coverage_preference
        ]):
            raise HTTPException(status_code=400, detail="Incomplete plan discovery information")
        
        # Search for eligible plans
        eligible_plans = search_eligible_plans(session.plan_discovery_answers)
        
        if not eligible_plans:
            return PlanAnalysisResponse(
                analysis="No eligible plans found for your business profile. Please contact us directly for assistance.",
                eligible_plans_count=0,
                session_id=session_id
            )
        
        # Analyze and rank the plans
        analysis_result = reason_about_plans(eligible_plans, session.plan_discovery_answers)
        
        return PlanAnalysisResponse(
            analysis=analysis_result,
            eligible_plans_count=len(eligible_plans),
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan analysis error: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "user_id": session.user_id,
        "chat_history_length": len(session.chat_history),
        "extracted_entities_count": len(session.extracted_entities),
        "plan_discovery_answers": session.plan_discovery_answers,
        "plan_discovery_complete": (
            session.plan_discovery_answers is not None and
            all([
                session.plan_discovery_answers.business_size,
                session.plan_discovery_answers.location,
                session.plan_discovery_answers.coverage_preference
            ])
        )
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys())
    }

# ==================== DATA PROCESSING ENDPOINTS ====================

@app.post("/data/scrape")
async def scrape_plans(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Scrape insurance plan data from provided URLs (or default URLs)
    BackgroundTasks allows the function to return immediately while work continues
    """
    job_id = create_job(request.job_name)
    
    # Use provided URLs or default plan_links
    urls = [str(url) for url in request.urls] if request.urls else plan_links
    
    background_tasks.add_task(_scrape_task, job_id, urls)
    
    return {"job_id": job_id, "status": "started", "message": f"Scraping {len(urls)} URLs"}

async def _scrape_task(job_id: str, urls: List[str]):
    """Background task for scraping and cleaning"""
    try:
        update_job_status(job_id, JobStatus.RUNNING, progress="Scraping and cleaning URLs...")
        
        # smart_scraper now handles scraping, cleaning, and MongoDB storage in one step
        cleaned_documents = smart_scraper.scrape_and_store_if_not_exists(urls)
        
        result = {
            "cleaned_count": len(cleaned_documents),
            "urls": urls,
            "note": "Documents are scraped, cleaned, and stored in MongoDB"
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, result=result)
        
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))

@app.post("/data/process")
async def process_plans(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process scraped data and upload to MongoDB
    Intelligently checks if insurance models exist and creates them if needed
    """
    job_id = create_job(request.job_name)
    
    background_tasks.add_task(_process_task, job_id)
    
    return {"job_id": job_id, "status": "started", "message": "Processing insurance plans"}

async def _process_task(job_id: str):
    """Background task for processing with intelligent MongoDB model checking"""
    try:
        update_job_status(job_id, JobStatus.RUNNING, progress="Loading scraped and cleaned data...")
        
        # Load scraped and cleaned data from MongoDB
        cleaned_data = smart_scraper.scrape_and_store_if_not_exists(plan_links)
        
        # Check if insurance models exist in MongoDB
        if not check_models_exist_in_mongodb():
            update_job_status(job_id, JobStatus.RUNNING, progress="Insurance models not found in MongoDB, analyzing plan fields...")
            plan_analysis(cleaned_data)
            models_existed = False
        else:
            update_job_status(job_id, JobStatus.RUNNING, progress="Using existing insurance models from MongoDB...")
            models_existed = True
        
        update_job_status(job_id, JobStatus.RUNNING, progress="Loading model definitions from MongoDB...")
        data = load_models_from_mongodb()
        
        DynamicInsurancePlanModel, DynamicMetaDataTags = generate_pydantic_models(
            required_fields=data["required_fields"],
            key_differences=data["key_differences"]
        )
        
        update_job_status(job_id, JobStatus.RUNNING, progress="Processing and uploading to MongoDB...")
        uploaded_count = process_pages_to_mongodb(cleaned_data)
        
        result = {
            "processed_count": len(cleaned_data),
            "uploaded_count": uploaded_count,
            "collection": "cigna_insurance.insurance_plans",
            "models_existed": models_existed
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, result=result)
        
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))

@app.post("/data/scrape-and-process")
async def scrape_and_process(request: ScrapeAndProcessRequest, background_tasks: BackgroundTasks):
    """
    Combined endpoint: scrape URLs then process and upload to MongoDB
    Most convenient for full automation
    """
    job_id = create_job(request.job_name)
    
    # Use provided URLs or default plan_links
    urls = [str(url) for url in request.urls] if request.urls else plan_links
    
    background_tasks.add_task(_scrape_and_process_task, job_id, urls)
    
    return {"job_id": job_id, "status": "started", "message": f"Full processing of {len(urls)} URLs"}

async def _scrape_and_process_task(job_id: str, urls: List[str]):
    """Combined scrape and process task"""
    try:
        # Step 1: Scrape and Clean (now combined)
        update_job_status(job_id, JobStatus.RUNNING, progress="Step 1: Scraping and cleaning URLs...")
        cleaned_data = smart_scraper.scrape_and_store_if_not_exists(urls)
        
        # Step 2: Check and create models if needed in MongoDB
        if not check_models_exist_in_mongodb():
            update_job_status(job_id, JobStatus.RUNNING, progress="Step 2: Analyzing plan fields...")
            plan_analysis(cleaned_data)
        
        # Step 3: Load models from MongoDB
        update_job_status(job_id, JobStatus.RUNNING, progress="Step 3: Loading model inputs from MongoDB...")
        data = load_models_from_mongodb()
        
        DynamicInsurancePlanModel, DynamicMetaDataTags = generate_pydantic_models(
            required_fields=data["required_fields"],
            key_differences=data["key_differences"]
        )
        
        # Step 4: Process and upload
        update_job_status(job_id, JobStatus.RUNNING, progress="Step 4: Processing and uploading to MongoDB...")
        uploaded_count = process_pages_to_mongodb(cleaned_data)
        
        result = {
            "scraped_and_cleaned_count": len(cleaned_data),
            "uploaded_count": uploaded_count,
            "urls": urls,
            "collection": "cigna_insurance.insurance_plans"
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, result=result)
        
    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))

@app.get("/data/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    job_info = get_job_info(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_info

@app.get("/data/jobs")
async def list_jobs(limit: int = 10):
    """List recent jobs"""
    jobs = list(jobs_store.values())
    jobs.sort(key=lambda x: x.updated_at, reverse=True)
    return {"jobs": jobs[:limit], "total": len(jobs)}

@app.get("/data/plans/count")
async def get_plans_count():
    """Get count of plans in MongoDB"""
    try:
        count = collection.count_documents({})
        summary_count = collection.count_documents({"summary": {"$exists": True}})
        return {
            "total_plans": count,
            "plans_with_summary": summary_count,
            "plans_without_summary": count - summary_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/data/health")
async def data_health_check():
    """Health check for data processing components"""
    health_status = {"api": "healthy"}
    
    # Check MongoDB
    try:
        mongo_client.admin.command('ping')
        health_status["mongodb"] = "connected"
        
        # Check insurance plans collection
        plans_count = collection.count_documents({})
        health_status["mongodb_plans_count"] = plans_count
        
        # Check models collection
        models_count = models_collection.count_documents({"model_type": "insurance_models"})
        health_status["mongodb_models_count"] = models_count
        health_status["models_in_mongodb"] = models_count > 0
        
        # Check scraped documents collection
        scraped_count = smart_scraper.scraped_collection.count_documents({})
        health_status["mongodb_scraped_count"] = scraped_count
        
    except Exception as e:
        health_status["mongodb"] = f"error: {str(e)}"
    
    # Check if local insurance models exist (for migration)
    models_path = Path("insurance_models.py")
    health_status["local_insurance_models"] = "exists" if models_path.exists() else "missing"
    
    return health_status

@app.post("/data/upload-local-models")
async def upload_local_models():
    """
    ONE-TIME USE ENDPOINT: Upload local insurance_models.py file to MongoDB
    Use this once to migrate from local file to MongoDB storage
    """
    try:
        models_id = upload_local_models_to_mongodb()
        return {
            "success": True,
            "message": "Successfully uploaded local models to MongoDB",
            "models_id": str(models_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading models: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)