from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import uuid


from insurance_agent import (
    SessionState, 
    ask_rag_bot, 
    plan_discovery_node,
    search_eligible_plans,
    reason_about_plans
)
from schemas import PlanDiscoveryAnswers

app = FastAPI(
    title="Cigna Insurance Chatbot API",
    description="FastAPI service for Cigna insurance chatbot functionality",
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

def create_session_id() -> str:
    """Create a new session ID"""
    return str(uuid.uuid4())

def get_session(session_id: str) -> SessionState:
    """Get existing session or raise error"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.get("/")
async def root():
    return {"message": "Cigna Insurance Chatbot API is running"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)