from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
from models.schemas import PlanDiscoveryAnswers


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


class ScrapeRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None  # If None, use default plan_links
    job_name: Optional[str] = None


class ProcessRequest(BaseModel):
    job_name: Optional[str] = None


class ScrapeAndProcessRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None  # If None, use default plan_links
    job_name: Optional[str] = None