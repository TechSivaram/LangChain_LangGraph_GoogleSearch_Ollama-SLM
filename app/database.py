# app/database.py
import sqlite3
import json
from typing import List, Dict, Any

DATABASE_FILE = "chat_history.db"

def initialize_db(): # Renamed from init_db to initialize_db
    """
    Initializes the SQLite database and creates the chat_sessions table if it doesn't exist.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                history TEXT
            )
        ''')
        conn.commit()
    print(f"Database initialized: {DATABASE_FILE}")

def save_chat_history(session_id: str, history: List[Dict[str, Any]]):
    """
    Saves or updates the chat history for a given session ID.
    History is stored as a JSON string.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        history_json = json.dumps(history)
        cursor.execute(
            "INSERT OR REPLACE INTO chat_sessions (session_id, history) VALUES (?, ?)",
            (session_id, history_json)
        )
        conn.commit()
    # print(f"History saved for session: {session_id}") # Uncomment for detailed logging

def load_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Loads and returns the chat history for a given session ID.
    Returns an empty list if no history is found.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history FROM chat_sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        if result:
            # print(f"History loaded for session: {session_id}") # Uncomment for detailed logging
            return json.loads(result[0])
        # print(f"No history found for session: {session_id}") # Uncomment for detailed logging
        return []

def get_all_session_ids() -> List[str]:
    """
    Retrieves a list of all unique session IDs stored in the database.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM chat_sessions")
        session_ids = [row[0] for row in cursor.fetchall()]
        # print(f"Retrieved {len(session_ids)} session IDs.") # Uncomment for detailed logging
        return session_ids

def delete_session_history(session_id: str) -> bool:
    """
    Deletes the chat history for a given session ID.
    Returns True if deleted, False if session not found.
    """
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Session history deleted for: {session_id}")
            return True
        print(f"No session found to delete for: {session_id}")
        return False

# This line ensures the database is initialized when the module is imported
initialize_db()