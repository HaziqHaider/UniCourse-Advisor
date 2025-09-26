from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import RAGEngine
from agent import IntelliCourseAgent
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="IntelliCourse API",
    description="AI-Powered University Course Advisor",
    version="1.0.0"
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_tool: str
    retrieved_context: str = ""

# Initialize components
rag_engine = RAGEngine()
agent = IntelliCourseAgent(rag_engine)

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    try:
        rag_engine.load_and_process_documents()
        print("RAG engine initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG engine: {e}")

@app.get("/")
async def root():
    return {"message": "IntelliCourse API is running!"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """Main endpoint for course advisor queries"""
    try:
        answer = agent.query(request.query)
        
        return QueryResponse(
            answer=answer,
            source_tool="intellicourse_agent",
            retrieved_context="Context retrieved from course database and/or web search"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)