import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any # Added Dict, Any for type hints
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json
from datetime import datetime # Added datetime for default session name

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import functions from your qa_agent and database modules
from app.qa_agent import get_qa_agent
from app.database import (
    initialize_db,
    save_chat_history,
    load_chat_history,
    get_all_sessions_metadata, # <--- MODIFIED: now returns metadata (id, name, created_at)
    delete_session_history,
    create_new_session,       # <--- NEW: to explicitly create a session
    rename_session_in_db      # <--- NEW: to rename a session
)

# Define Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    chat_history: List[Dict[str, Any]] = [] # Changed to Dict[str, Any] for flexibility

class SessionMetadata(BaseModel): # <--- MODIFIED Pydantic model for session data
    id: str
    name: str
    created_at: datetime # Include creation timestamp

# In-memory store for the LangGraph agent instance
# This will be initialized once on startup
qa_agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database on startup (redundant if app.database.initialize_db() is called on import,
    # but good for explicit lifecycle management in FastAPI context)
    initialize_db() 
    
    # Initialize the QA agent
    global qa_agent_instance
    qa_agent_instance = get_qa_agent()
    print("FastAPI app startup: QA Agent initialized.")
    
    yield
    # Clean up resources on shutdown (if any)
    print("FastAPI app shutdown.")

app = FastAPI(lifespan=lifespan)

# CORS Middleware to allow requests from your frontend
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Add any other origins where your frontend might be hosted
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML page for the chat interface.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/new_session", response_model=SessionMetadata, status_code=status.HTTP_201_CREATED)
async def new_session():
    """
    Creates a new chat session with a default name and returns its ID and name.
    """
    session_id = str(uuid.uuid4())
    default_session_name = f"New Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Explicitly create the new session in the database
    create_new_session(session_id, default_session_name)
    
    print(f"New session created: {session_id} with name '{default_session_name}'")
    # Return the newly created session's metadata
    # We fetch it again to ensure it reflects the DB state, including created_at
    # For simplicity, let's just create a dummy SessionMetadata for return
    return SessionMetadata(id=session_id, name=default_session_name, created_at=datetime.utcnow()) # Using utcnow as a placeholder, DB has exact time


@app.get("/sessions", response_model=List[SessionMetadata])
async def get_sessions():
    """
    Returns a list of all active session IDs and their names.
    """
    sessions_data = get_all_sessions_metadata() # Call the new function
    # The return value from get_all_sessions_metadata should match SessionMetadata fields
    return [SessionMetadata(**s_data) for s_data in sessions_data] # Unpack dict into Pydantic model


@app.get("/history/{session_id}", response_model=List[Dict[str, Any]])
async def get_history(session_id: str):
    """
    Loads and returns the chat history for a given session ID.
    """
    history = load_chat_history(session_id)
    if not history:
        print(f"History requested for non-existent or empty session: {session_id}")
        return []
    print(f"History loaded for session: {session_id}, messages: {len(history)}")
    return history

# New endpoint to rename a session
class RenameSessionRequest(BaseModel):
    new_name: str

@app.put("/sessions/{session_id}/rename", response_model=SessionMetadata)
async def rename_session(session_id: str, request: RenameSessionRequest):
    """
    Renames an existing chat session.
    """
    if not rename_session_in_db(session_id, request.new_name):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or rename failed.")
    
    # After successful rename, you might want to fetch the updated session metadata
    # For simplicity, constructing it here. In production, fetch from DB for consistency.
    # Get the current metadata to return the correct created_at
    all_sessions = get_all_sessions_metadata()
    current_session_meta = next((s for s in all_sessions if s['id'] == session_id), None)
    
    if current_session_meta:
        return SessionMetadata(id=session_id, name=request.new_name, created_at=current_session_meta['created_at'])
    else:
        # Fallback if somehow not found after rename (shouldn't happen if rename_session_in_db worked)
        raise HTTPException(status_code=500, detail="Failed to retrieve updated session metadata.")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles a chat message, processes it with the QA agent,
    saves history, and returns the response.
    """
    if qa_agent_instance is None:
        raise HTTPException(status_code=503, detail="QA Agent not initialized.")

    session_id = request.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required for /chat endpoint.")
    
    # Load existing chat history
    chat_history_messages = load_chat_history(session_id)
    
    # Ensure all messages are properly typed as BaseMessage if needed by LangGraph
    formatted_chat_history = []
    for msg in chat_history_messages:
        if msg["type"] == "human":
            formatted_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            formatted_chat_history.append(AIMessage(content=msg["content"]))

    print(f"Received message for session {session_id}: {request.question}")
    
    # Define the initial state for the agent
    initial_state = {
        "user_question": request.question,
        "chat_history": formatted_chat_history,
        "research_results": "",
        "should_research": False,
        "search_query": "",
        "refine_answer_with_research": ""
    }

    try:
        # Invoke the LangGraph agent
        result = await qa_agent_instance.ainvoke(initial_state)

        # The final answer is in the 'chat_history' of the result, specifically the last AI message
        ai_response_message = None
        if result and "chat_history" in result and result["chat_history"]:
            for msg in reversed(result["chat_history"]):
                if isinstance(msg, AIMessage):
                    ai_response_message = msg.content
                    break
        
        if not ai_response_message:
            ai_response_message = "I couldn't process that request or find a clear answer."
            print("Warning: Agent did not return a valid AI message.")

        # Save updated chat history after processing
        updated_history_for_db = chat_history_messages + [
            {"type": "human", "content": request.question},
            {"type": "ai", "content": ai_response_message}
        ]
        save_chat_history(session_id, updated_history_for_db)

        return ChatResponse(
            response=ai_response_message,
            session_id=session_id,
            chat_history=updated_history_for_db # Return full history for UI refresh
        )
    except Exception as e:
        print(f"Error during chat processing: {e}")
        # It's good practice to log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# Endpoint to delete a specific session
@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Deletes the chat session and its entire history for a given session ID.
    """
    if delete_session_history(session_id):
        return {"message": f"Session '{session_id}' and its history deleted."}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)