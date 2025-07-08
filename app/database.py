# app/database.py
import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime

DATABASE_FILE = "chat_history.db"

def _get_db_connection():
    return sqlite3.connect(DATABASE_FILE)

def initialize_db():
    """
    Initializes the SQLite database and creates/alters the chat_sessions table.
    Ensures 'name' and 'created_at' columns exist. Handles migrations for existing tables.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()

        # Create table if it doesn't exist with all columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT DEFAULT 'New Chat',
                history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

        # --- Migration logic for existing tables without new columns ---
        cursor.execute("PRAGMA table_info(chat_sessions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add 'name' column if missing
        if 'name' not in columns:
            print("Adding 'name' column to chat_sessions table...")
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN name TEXT DEFAULT 'New Chat'")
            conn.commit()
            print("Successfully added 'name' column.")
        
        # Add 'created_at' column if missing (handles the default CURRENT_TIMESTAMP error)
        if 'created_at' not in columns:
            print("Adding 'created_at' column to chat_sessions table...")
            # Step 1: Add column allowing NULL (SQLite constraint for ALTER TABLE ADD COLUMN)
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN created_at TIMESTAMP")
            conn.commit()
            
            # Step 2: Update existing rows with a default timestamp
            cursor.execute("UPDATE chat_sessions SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
            conn.commit()
            
            # Step 3 (Optional, but good): Add NOT NULL constraint and DEFAULT for future inserts
            # SQLite does not directly support ALTER COLUMN ADD DEFAULT or ALTER COLUMN SET NOT NULL
            # This is typically handled by creating a new table and copying data if strict constraints are needed
            # For simplicity, we'll just rely on the Python code inserting CURRENT_TIMESTAMP for new sessions.
            # The DEFAULT CURRENT_TIMESTAMP in the CREATE TABLE IF NOT EXISTS above will apply to truly new tables.
            
            print("Successfully added and populated 'created_at' column.")

    print(f"Database initialized and schema checked: {DATABASE_FILE}")

def create_new_session(session_id: str, name: str):
    """
    Creates a new session entry in the database with a given ID and name.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        # Insert with history as empty JSON array and current timestamp
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, name, history, created_at) VALUES (?, ?, ?, ?)",
            (session_id, name, json.dumps([]), datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')) # Use strftime for consistent timestamp format
        )
        conn.commit()
    print(f"Created new session record: {session_id} - '{name}'")


def save_chat_history(session_id: str, history: List[Dict[str, Any]]):
    """
    Saves or updates the chat history for a given session ID.
    History is stored as a JSON string.
    NOTE: This function assumes the session_id already exists in the table.
          Use create_new_session first for new sessions.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        history_json = json.dumps(history)
        cursor.execute(
            "UPDATE chat_sessions SET history = ? WHERE session_id = ?",
            (history_json, session_id)
        )
        conn.commit()
    # print(f"History saved for session: {session_id}")

def load_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Loads and returns the chat history for a given session ID.
    Returns an empty list if no history is found.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history FROM chat_sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return []

def get_all_sessions_metadata() -> List[Dict[str, Any]]:
    """
    Retrieves a list of all sessions including their ID, name, and creation timestamp.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, name, created_at FROM chat_sessions ORDER BY created_at DESC")
        sessions_data = []
        for row in cursor.fetchall():
            session_id, name, created_at_str = row
            # Convert string to datetime object
            try:
                # Handle both with and without microseconds in string
                created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
            sessions_data.append({"id": session_id, "name": name, "created_at": created_at})
        return sessions_data

def rename_session_in_db(session_id: str, new_name: str) -> bool:
    """
    Renames a session in the database.
    Returns True if renamed, False if session not found.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE chat_sessions SET name = ? WHERE session_id = ?", (new_name, session_id))
        conn.commit()
        return cursor.rowcount > 0

def delete_session_history(session_id: str) -> bool:
    """
    Deletes the chat session and its entire history for a given session ID.
    Returns True if deleted, False if session not found.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Session '{session_id}' and its history deleted.")
            return True
        print(f"No session found to delete for: {session_id}")
        return False

# Ensure database is initialized when this module is imported
initialize_db()