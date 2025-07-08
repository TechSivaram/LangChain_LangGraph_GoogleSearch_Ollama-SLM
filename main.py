import os
from contextlib import asynccontextmanager
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # <--- ADDED AIMessage and BaseMessage here!

# Import functions from your qa_agent and database modules
from app.qa_agent import get_qa_agent
from app.database import (
    initialize_db, save_chat_history, load_chat_history, get_all_session_ids, delete_session_history
)

# Define Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    chat_history: List[dict] = [] # Include chat_history in response

class SessionResponse(BaseModel):
    session_id: str

# In-memory store for the LangGraph agent instance
# This will be initialized once on startup
qa_agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database on startup
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

@app.post("/new_session", response_model=SessionResponse)
async def new_session():
    """
    Creates a new chat session and returns its ID.
    """
    session_id = str(uuid.uuid4())
    # Optionally, you could initialize an empty history for the new session here
    # save_chat_history(session_id, [])
    print(f"New session created: {session_id}")
    return {"session_id": session_id}

@app.get("/sessions", response_model=List[str])
async def get_sessions():
    """
    Returns a list of all active session IDs.
    """
    sessions = get_all_session_ids()
    return sessions

@app.get("/history/{session_id}", response_model=List[dict])
async def get_history(session_id: str):
    """
    Loads and returns the chat history for a given session ID.
    """
    history = load_chat_history(session_id)
    if not history:
        # If history is empty, return 200 OK with an empty list, not 404
        # A 404 would typically be for a resource that doesn't exist *at all*
        # But an empty history for a session is valid.
        # If the session_id itself is truly non-existent in the DB, you might
        # still want to raise 404, but for now, an empty list is sufficient.
        print(f"History requested for non-existent or empty session: {session_id}")
        # Raising HTTPException(status_code=404) here if the session_id is genuinely not in your list
        # of known session IDs from `get_all_session_ids()` might be more accurate if the UI needs it.
        # For simplicity, returning an empty list might be handled gracefully by UI.
        # For the previous 404, it was likely due to the ID not being found in DB from the `load_chat_history`.
        # Ensure `load_chat_history` truly handles non-existent IDs gracefully by returning `[]`.
        return []
    print(f"History loaded for session: {session_id}, messages: {len(history)}")
    return history

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles a chat message, processes it with the QA agent,
    saves history, and returns the response.
    """
    if qa_agent_instance is None:
        raise HTTPException(status_code=503, detail="QA Agent not initialized.")

    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    
    # Load existing chat history
    chat_history_messages = load_chat_history(session_id)
    
    # Ensure all messages are properly typed as BaseMessage if needed by LangGraph
    # For now, assuming they are dicts from DB, LangGraph will convert them
    # For simplicity, convert all history entries to HumanMessage/AIMessage objects
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
        # Use .ainvoke for async execution
        result = await qa_agent_instance.ainvoke(initial_state)

        # The final answer is in the 'chat_history' of the result, specifically the last AI message
        ai_response_message = None
        if result and "chat_history" in result and result["chat_history"]:
            # Iterate backwards to find the last AIMessage
            for msg in reversed(result["chat_history"]):
                if isinstance(msg, AIMessage):
                    ai_response_message = msg.content
                    break
        
        if not ai_response_message:
            ai_response_message = "I couldn't process that request."
            print("Warning: Agent did not return a valid AI message.")

        # Save updated chat history after processing
        # Append user question and AI response to history before saving
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
        # Optionally, save the error message to history or log more details
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# Endpoint to delete session history (useful for development)
@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """
    Deletes the chat history for a given session ID.
    """
    if delete_session_history(session_id):
        return {"message": f"Session {session_id} history deleted."}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)