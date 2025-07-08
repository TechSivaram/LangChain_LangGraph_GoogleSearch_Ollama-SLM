import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone # Import timezone
import uuid # For generating UUIDs for session IDs

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from pydantic import BaseModel, EmailStr # EmailStr might not be used but good to have
from jose import JWTError, jwt
from passlib.context import CryptContext

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import functions/models from your local modules
from app.qa_agent import get_qa_agent
from app.database import create_db_and_tables, get_db, User, Session as DBSession, Message as DBMessage # Renamed Session to DBSession to avoid conflict with sqlalchemy.orm.Session


# --- Security Configuration ---
SECRET_KEY = os.environ.get("SECRET_KEY", "your-super-secure-jwt-key-replace-me-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login") # Token URL for OAuth2

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# --- Pydantic Models for API Requests/Responses ---
class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True # Allows Pydantic to read from SQLAlchemy models

class Token(BaseModel):
    access_token: str
    token_type: str

class MessageBase(BaseModel):
    type: str
    content: str
    created_at: datetime

class SessionMetadata(BaseModel):
    id: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None # Still optional for frontend, but backend will ensure it exists

class ChatResponse(BaseModel):
    response: str
    session_id: str
    chat_history: List[MessageBase] # Changed to MessageBase

class RenameSessionRequest(BaseModel):
    new_name: str


# In-memory store for the LangGraph agent instance
qa_agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database and tables on startup
    create_db_and_tables()
    
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


# --- User Authentication Endpoints ---
@app.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    
    hashed_password = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, password_hash=hashed_password)
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Registration failed: {e}")

@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Returns details of the currently authenticated user.
    """
    return current_user


# --- Session Management Endpoints (User-Isolated) ---
@app.post("/new_session", response_model=SessionMetadata, status_code=status.HTTP_201_CREATED)
async def new_session_for_user(
    current_user: User = Depends(get_current_user), # Requires authentication
    db: Session = Depends(get_db)
):
    """
    Creates a new chat session linked to the current authenticated user.
    """
    session_id = str(uuid.uuid4())
    
    # Removed the line below which was generating the timestamp in the name:
    # default_session_name = f"New Chat - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    
    db_session = DBSession(
        id=session_id,
        user_id=current_user.id, # Link session to the authenticated user
        name="New Chat" # Explicitly set the name to "New Chat"
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    print(f"New session created for user {current_user.username}: {session_id} with name '{db_session.name}'")
    return db_session


@app.get("/sessions", response_model=List[SessionMetadata])
async def get_user_sessions(
    current_user: User = Depends(get_current_user), # Requires authentication
    db: Session = Depends(get_db)
):
    """
    Returns a list of all chat sessions belonging to the current authenticated user.
    """
    # Filter sessions by current user's ID
    sessions = db.query(DBSession).filter(DBSession.user_id == current_user.id).order_by(DBSession.created_at.desc()).all()
    return sessions


@app.get("/history/{session_id}", response_model=List[MessageBase])
async def get_chat_history_for_session(
    session_id: str,
    current_user: User = Depends(get_current_user), # Requires authentication
    db: Session = Depends(get_db)
):
    """
    Loads and returns the chat history for a specific session,
    ensuring it belongs to the current user.
    """
    # First, check if the session exists and belongs to the current user
    session = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized for this user.")

    # Load messages associated with this session
    messages = db.query(DBMessage).filter(DBMessage.session_id == session_id).order_by(DBMessage.created_at).all()
    
    print(f"History loaded for session: {session_id}, messages: {len(messages)}")
    # Map DB Message objects to Pydantic MessageBase
    return [
        MessageBase(type=msg.type, content=msg.content, created_at=msg.created_at)
        for msg in messages
    ]


@app.put("/sessions/{session_id}/rename", response_model=SessionMetadata)
async def rename_user_session(
    session_id: str,
    rename_request: RenameSessionRequest,
    current_user: User = Depends(get_current_user), # Requires authentication
    db: Session = Depends(get_db)
):
    """
    Renames an existing chat session, ensuring it belongs to the current user.
    """
    session = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized for this user.")

    session.name = rename_request.new_name
    db.commit()
    db.refresh(session)
    return session


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_session(
    session_id: str,
    current_user: User = Depends(get_current_user), # Requires authentication
    db: Session = Depends(get_db)
):
    """
    Deletes a specific chat session and its history, ensuring it belongs to the current user.
    """
    session = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized for this user.")

    db.delete(session) # SQLAlchemy's cascade="all, delete-orphan" will delete associated Messages
    db.commit()
    print(f"Session '{session_id}' and its history deleted for user {current_user.username}.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/chat", response_model=ChatResponse)
async def chat_with_ollama_user_isolated(
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_user), # Requires authentication
    db: Session = Depends(get_db)
):
    """
    Handles a chat message, processes it with the QA agent,
    saves history, and returns the response, ensuring session belongs to the user.
    """
    if qa_agent_instance is None:
        raise HTTPException(status_code=503, detail="QA Agent not initialized.")

    session_id = chat_request.session_id
    question = chat_request.question

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required for /chat endpoint.")

    # Ensure the session belongs to the current user
    session = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized for this user.")

    # Load existing chat history from DB for this session
    db_messages = db.query(DBMessage).filter(DBMessage.session_id == session_id).order_by(DBMessage.created_at).all()
    
    # Convert DB messages to LangChain BaseMessage format
    langchain_history: List[BaseMessage] = []
    for msg in db_messages:
        if msg.type == 'human':
            langchain_history.append(HumanMessage(content=msg.content))
        elif msg.type == 'ai':
            langchain_history.append(AIMessage(content=msg.content))

    print(f"User {current_user.username} - Session {session_id}: Received message: {question}")

    # Save user's question to DB
    user_message = DBMessage(session_id=session_id, type="human", content=question)
    db.add(user_message)
    db.commit()
    db.refresh(user_message) # Refresh to get created_at timestamp

    # Prepare initial state for LangGraph
    initial_state = {
        "user_question": question,
        "chat_history": langchain_history,
        "research_results": "", # Initialize as empty
        "should_research": False, # Initialize
        "search_query": "", # Initialize
        "refine_answer_with_research": "" # Initialize
    }

    ai_response_content = "An error occurred while processing your request."
    try:
        # Invoke the LangGraph agent
        final_state = await qa_agent_instance.ainvoke(initial_state)

        # The final answer is the last AI message in the returned chat_history
        if "refine_answer_with_research" in final_state and final_state["refine_answer_with_research"]:
            ai_response_content = final_state["refine_answer_with_research"]
        elif final_state and "chat_history" in final_state and final_state["chat_history"]:
            for msg in reversed(final_state["chat_history"]):
                if isinstance(msg, AIMessage):
                    ai_response_content = msg.content
                    break
        else:
            # Fallback if the structure is unexpected
            ai_response_content = "I couldn't process that request or find a clear answer."
            print("Warning: LangGraph agent did not return a valid AI message in final_state.")

    except Exception as e:
        print(f"Error during chat processing for user {current_user.username}, session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        ai_response_content = f"An internal server error occurred while processing your request: {e}"

    # Save AI's response to DB
    ai_message = DBMessage(session_id=session_id, type="ai", content=ai_response_content)
    db.add(ai_message)
    db.commit()
    db.refresh(ai_message) # Refresh to get created_at timestamp

    # Reload full history for response to ensure timestamps are present and consistent
    updated_db_messages = db.query(DBMessage).filter(DBMessage.session_id == session_id).order_by(DBMessage.created_at).all()
    
    return ChatResponse(
        response=ai_response_content,
        session_id=session_id,
        chat_history=[
            MessageBase(type=msg.type, content=msg.content, created_at=msg.created_at)
            for msg in updated_db_messages
        ]
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)