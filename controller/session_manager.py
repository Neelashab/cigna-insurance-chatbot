import uuid
from typing import Optional, Dict
from datetime import datetime
from fastapi import HTTPException

from controller.insurance_agent import SessionState
from models.api_models import JobInfo, JobStatus


# In-memory session storage (in production, use Redis)
sessions: Dict[str, SessionState] = {}

# Global job store (in production, use Redis or database)
jobs_store: Dict[str, JobInfo] = {}


def create_session_id() -> str:
    """Create a new session ID"""
    return str(uuid.uuid4())


def get_session(session_id: str) -> SessionState:
    """Get existing session or raise error"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


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