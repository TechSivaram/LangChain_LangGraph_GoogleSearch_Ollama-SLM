import os
import sqlite3
import uuid # For generating unique session IDs
from typing import TypedDict, Annotated, List, Literal
from langchain_ollama import ChatOllama
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from datetime import datetime
import pytz
from pydantic import BaseModel, Field

# --- Load environment variables from .env file ---
from dotenv import load_dotenv
load_dotenv()

# --- 1. Environment Variables Check and Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3") # Default to phi3, but llama3 is recommended for better reliability

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please set it in your .env file. "
        "Get it from Google Cloud Console."
    )
if not GOOGLE_CSE_ID:
    raise ValueError(
        "GOOGLE_CSE_ID not found. Please set it in your .env file. "
        "Get it from Google Programmable Search Engine."
    )
if not OLLAMA_MODEL:
    raise ValueError(
        "OLLAMA_MODEL not found. Please set it in your .env file (e.g., OLLAMA_MODEL=\"phi3\")."
    )

print(f"--- Configuration ---")
print(f"Ollama Model: {OLLAMA_MODEL}")
print(f"Google API Key loaded: {'YES' if GOOGLE_API_KEY else 'NO'}")
print(f"Google CSE ID loaded: {'YES' if GOOGLE_CSE_ID else 'NO'}")
print(f"---------------------")

# --- SQLite Database Configuration ---
DB_FILE = "chat_history.db"

def init_db():
    """Initializes the SQLite database and creates the chat_messages table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"--- Database '{DB_FILE}' initialized. ---")

def save_message(session_id: str, role: Literal["human", "ai"], content: str):
    """Saves a single message to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now(pytz.utc).isoformat()
    cursor.execute(
        "INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, timestamp)
    )
    conn.commit()
    conn.close()
    # print(f"--- DEBUG: Message saved for session '{session_id}': {role} - {content[:50]}... ---")

def load_history(session_id: str) -> List[BaseMessage]:
    """Loads chat history for a given session_id from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    )
    messages = []
    for row in cursor.fetchall():
        role, content = row
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    conn.close()
    print(f"--- DEBUG: Loaded {len(messages)} messages for session '{session_id}'. ---")
    return messages

# --- 2. Define LangChain Tool for Google Search ---
@tool
def search_tool(query: str) -> str:
    """
    Performs a web search using Google Search to find information.
    Useful when you need to research a topic or find current, factual information.
    Input should be a clear, concise search query string.
    """
    print(f"\n--- DEBUG: NODE - Performing Google Search for query: '{query}' ---")
    try:
        search = GoogleSearchAPIWrapper(k=5) # Request top 5 results for better focus
        results = search.run(query)
        print(f"--- DEBUG: Search Results: {results}")
        return results
    except Exception as e:
        error_msg = (f"Error performing Google Search for '{query}': {e}. "
                     "Ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are correct, "
                     "the Custom Search API is enabled in your Google Cloud Project, "
                     "and billing is enabled if you've exceeded the free tier.")
        print(f"--- ERROR: {error_msg} ---")
        return error_msg

# --- 3. Define the LangGraph State ---
class AgentState(TypedDict):
    """
    Represents the state of our agent in LangGraph.
    - messages: A list of messages (conversation history). `Annotated` with `lambda x, y: x + y`
                tells LangGraph to append new messages to the existing list while preserving history.
    - search_query: The query string used for web search (set by `decide_to_research`).
    - research_results: The text results obtained from the web search (set by `conduct_research`).
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    search_query: str
    research_results: str

# --- 4. Initialize Ollama SLM (LangChain Integration) ---
try:
    print(f"--- Initializing Ollama Chat Model: {OLLAMA_MODEL} ---")
    slm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
except Exception as e:
    raise RuntimeError(
        f"Failed to initialize Ollama Chat model '{OLLAMA_MODEL}'. "
        f"Please ensure Ollama is running (`ollama serve` or desktop app) "
        f"and the model is pulled (`ollama pull {OLLAMA_MODEL}`). Error: {e}"
    )

# --- Define Pydantic output for structured decision ---
class ResearchDecision(BaseModel):
    """
    Decision on whether external research is needed and the search query if so.
    """
    should_research: bool = Field(
        description="True if external web research is necessary to provide an accurate, current, or comprehensive answer. False otherwise."
    )
    search_query: str = Field(
        description="A concise and effective search query string if research is needed. Empty string if not needed.",
        default=""
    )

# --- 5. Define LangGraph Nodes (Functions) ---
def call_slm_initial(state: AgentState):
    """
    Node: Calls the Ollama SLM to generate an initial answer based on the user's question.
    """
    print("\n--- NODE: Calling SLM for Initial Answer ---")
    current_messages = state['messages']

    response = slm.invoke(current_messages)

    print(f"--- DEBUG: SLM Initial Response: {response.content}")
    return {"messages": [response]}

def decide_to_research(state: AgentState):
    """
    Node: Determines if the SLM's initial answer is sufficient or if external web research is needed.
    This decision is made by prompting the SLM itself, using structured output for reliability.
    """
    print("\n--- NODE: Deciding if Research is Needed ---")

    # Get current time and location
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone)
    current_date_str = current_time_ist.strftime("%A, %B %d, %Y")
    current_time_str = current_time_ist.strftime("%I:%M:%S %p %Z")

    current_location = "Tenali, Andhra Pradesh, India" # Hardcoded for now

    # NEW: More Assertive Prompt for search
    decision_prompt_content = f"""
    **Current Context for Decision Making:**
    Today's Date: {current_date_str}
    Current Time: {current_time_str}
    Current Location: {current_location}

    You are an AI assistant specialized in providing the most accurate, comprehensive, and **strictly up-to-date** information.
    **Your internal knowledge is limited by its training data cutoff and is NOT guaranteed to be current.**

    Based on the user's question and the conversation history, you **MUST** determine if a web search is necessary.

    **It is HIGHLY PROBABLE and CRUCIAL to perform a web search if:**
    - The question explicitly asks for "latest," "current," "newest," "recent," or information "as of [specific date]".
    - The question relates to any topic that can change over time: news, ongoing events, political leaders, sports scores, scientific breakthroughs, technological developments, economic data, demographics (e.g., population), company information, or any real-time data (e.g., weather, stock prices).
    - If there is **ANY doubt** that your internal knowledge is perfectly up-to-date or sufficient for the current context.

    **Only set 'should_research' to false if the question is about static, well-established, historical facts that are extremely unlikely to change or be updated (e.g., "What is the capital of France?", "Who invented the lightbulb?", "What is 2+2?").**
    For all other questions, **assume a web search is needed** to provide the best, most current answer.

    If a web search is required, provide a concise and effective search query. The query should be optimized for a search engine to get the most relevant and up-to-date results.

    Provide your decision in the following JSON format:
    """

    decision_chain = slm.with_structured_output(ResearchDecision)

    messages_for_decision = [
        HumanMessage(content=decision_prompt_content)
    ] + state['messages']

    try:
        decision_response: ResearchDecision = decision_chain.invoke(messages_for_decision)
        print(f"--- DEBUG: SLM's Structured Research Decision RAW: {decision_response.model_dump_json()} ---")

        if decision_response.should_research and decision_response.search_query.strip():
            clean_query = decision_response.search_query.strip('"\' ')
            if len(clean_query.split()) >= 3 and len(clean_query) <= 250:
                print(f"--- DEBUG: Research IS needed. Search Query: '{clean_query}' ---")
                return {"search_query": clean_query}
            else:
                print(f"--- DEBUG: Research requested, but search_query was invalid (too short/long or malformed). Skipping search. Query: '{clean_query}' ---")
                return {"search_query": None}
        else:
            print(f"--- DEBUG: SLM decided NO RESEARCH IS NEEDED. ---")
            return {"search_query": None}

    except Exception as e:
        print(f"--- ERROR: Failed to get structured research decision from SLM: {e} ---")
        print("--- DEBUG: Attempting manual parsing fallback (less reliable)... ---")
        raw_response = slm.invoke(messages_for_decision).content.strip()
        print(f"--- DEBUG: Raw SLM response for decision (fallback): '{raw_response}' ---")
        if "no_research_needed" in raw_response.lower() or not raw_response.strip():
            return {"search_query": None}
        else:
            clean_query = raw_response.strip('"\' ')
            if len(clean_query.split()) >= 3 and len(clean_query) <= 250:
                print(f"--- DEBUG: Fallback found potential query: '{clean_query}' ---")
                return {"search_query": clean_query}
            else:
                print(f"--- DEBUG: Fallback: Response not a valid query. Skipping search. Response: '{clean_query}' ---")
                return {"search_query": None}

def conduct_research(state: AgentState):
    """
    Node: Executes the Google Search tool using the `search_query` from the state.
    """
    print("\n--- NODE: Conducting Research ---")
    search_query = state.get('search_query')

    if not search_query:
        print("--- DEBUG: No valid search query provided to conduct research. Skipping. ---")
        return {"research_results": "No valid search query provided by decision agent."}

    results = search_tool.invoke({"query": search_query})

    if "error performing google search" in results.lower():
        print(f"--- ERROR: Google Search returned an error: {results} ---")

    print(f"--- DEBUG: Search Results Received: {results}")
    return {"research_results": results}

def refine_answer_with_research(state: AgentState):
    """
    Node: Uses the Ollama SLM to refine the initial answer using the research results.
    """
    print("\n--- NODE: Refining Answer with Research Results ---")
    messages = state['messages']
    research_results = state['research_results']

    original_question = next((msg.content for msg in messages if isinstance(msg, HumanMessage)), "the user's original question")

    refinement_prompt_content = f"""
    You are an expert information synthesiser. Your task is to provide the most accurate, comprehensive, and up-to-date answer to the user's question, using the provided web search results.

    The user originally asked: "{original_question}"

    Your previous attempt to answer was: "{messages[-1].content}"

    Here are the web search results. Carefully read and extract relevant facts, figures, and recent information to answer the user's question.
    ---
    {research_results}
    ---

    **Instructions for your refined answer:**
    1.  **Prioritize the search results:** Use the information from the `research_results` as the primary source for factual details.
    2.  **Integrate and synthesize:** Combine relevant points from various search snippets into a coherent and well-structured answer. Do not just list snippets.
    3.  **Address the original question directly:** Ensure your answer fully addresses what the user asked.
    4.  **Conciseness and Clarity:** Be as concise as possible while providing a complete answer. Use clear and plain language.
    5.  **Handle Insufficiency/Contradictions:**
        * If the search results are not helpful, contain insufficient information, or contradict each other, state that fact clearly (e.g., "The search results did not provide specific information on X, but generally Y is true...") and then provide the best answer you can based on your knowledge or the limited results.
        * Do NOT invent information.
    6.  **No self-referencing:** Do not mention that you used a search tool, "research results," or "search snippets" unless the user explicitly asks for sources or how you found the information. Just provide the refined answer.
    7.  **Do NOT include links:** Your role is to provide the synthesized information directly, not to tell the user to visit links. The `research_results` might contain links, but you should extract text facts, not reproduce URLs.

    Based on the above, provide your refined, final answer:
    """

    slm_input_messages = messages + [HumanMessage(content=refinement_prompt_content)]

    response = slm.invoke(slm_input_messages)
    print(f"--- DEBUG: SLM Refined Response: {response.content}")
    return {"messages": [response]}

# --- 6. Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)

workflow.add_node("call_slm_initial", call_slm_initial)
workflow.add_node("decide_to_research", decide_to_research)
workflow.add_node("conduct_research", conduct_research)
workflow.add_node("refine_answer_with_research", refine_answer_with_research)

workflow.set_entry_point("call_slm_initial")

workflow.add_conditional_edges(
    "decide_to_research",
    lambda state: "research_needed" if state.get('search_query') else "answer_complete",
    {
        "research_needed": "conduct_research",
        "answer_complete": END
    }
)

workflow.add_edge("call_slm_initial", "decide_to_research")
workflow.add_edge("conduct_research", "refine_answer_with_research")
workflow.add_edge("refine_answer_with_research", END)

app = workflow.compile()

# --- 7. Run the Application ---

print("--- Smart Q&A System with Ollama SLM & Google Search ---")
print(f"Using Ollama Model: '{OLLAMA_MODEL}'")
print("Ensure Ollama is running and the specified model is pulled.")
print("Type 'exit' to quit.")

# Initialize the database
init_db()

current_session_id = None
chat_history: List[BaseMessage] = [] # Initialize here, will be populated after session choice

while True:
    if current_session_id is None:
        action = input("\nStart a (n)ew session or (l)oad existing? (n/l): ").lower().strip()
        if action == 'n':
            current_session_id = str(uuid.uuid4())
            print(f"--- New session started. Your session ID: {current_session_id} ---")
            chat_history = [] # Explicitly clear for new session
        elif action == 'l':
            session_to_load = input("Enter session ID to load: ").strip()
            if session_to_load:
                current_session_id = session_to_load
                chat_history = load_history(current_session_id)
                if not chat_history:
                    print(f"--- No history found for session ID: {current_session_id}. Starting fresh. ---")
            else:
                print("--- Invalid session ID. Starting a new session. ---")
                current_session_id = str(uuid.uuid4())
                chat_history = []
        else:
            print("Invalid choice. Please enter 'n' or 'l'.")
            continue # Loop back to ask again

    user_question = input(f"\n[{current_session_id[:8]}...] Your Question (type 'exit' to quit, 'new' to start new chat): ")
    if user_question.lower() == 'exit':
        print("Exiting application.")
        break
    elif user_question.lower() == 'new':
        print("--- Ending current session. ---")
        current_session_id = None # Reset session ID to prompt for new/load next loop
        continue # Loop back to session choice

    # Save user message to DB
    save_message(current_session_id, "human", user_question)

    # Add the current user question to the chat history for the current graph invocation
    chat_history.append(HumanMessage(content=user_question))

    initial_state = {"messages": chat_history, "search_query": "", "research_results": ""}

    try:
        print(f"\n--- Processing user question: '{user_question}' ---")
        final_state_from_invoke = app.invoke(initial_state)

        if final_state_from_invoke and 'messages' in final_state_from_invoke and final_state_from_invoke['messages']:
            ai_response = final_state_from_invoke['messages'][-1].content
            print(f"\nAI's Final Answer: {ai_response}")
            # Add the AI's response to the chat history and save to DB
            chat_history.append(AIMessage(content=ai_response))
            save_message(current_session_id, "ai", ai_response)
        else:
            print("\nAI could not generate a final answer. This likely means the graph ended without a final message. Check debug logs above for flow.")
            # If no final message, remove the last human message to avoid confusion
            if chat_history and isinstance(chat_history[-1], HumanMessage) and chat_history[-1].content == user_question:
                chat_history.pop() # Remove the unanswered human message from in-memory history


    except Exception as e:
        print(f"\n--- CRITICAL ERROR DURING APPLICATION RUN ---")
        print(f"An unhandled exception occurred: {e}")
        print("\n--- Common Troubleshooting Steps ---")
        print("1. **Ollama Server:** Is Ollama running? Open a terminal and type `ollama serve` or start the desktop app.")
        print(f"2. **Ollama Model:** Is the model '{OLLAMA_MODEL}' pulled? Run `ollama list` and `ollama pull {OLLAMA_MODEL}`.")
        print("   -> **Recommendation:** For better tool use, consider `llama3` if `phi3` struggles.")
        print("3. **Google API Keys:** Check your `.env` file for `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`. Are they correct?")
        print("   - Did you enable the Custom Search API in Google Cloud Console?")
        print("   - Did you create a Programmable Search Engine and link it to 'search the entire web'?")
        print("4. **Dependencies:** Did you run `pip install -r requirements.txt` successfully in your project's virtual environment?")
        print("   - Specifically check for `pydantic` and `langchain-core` versions (`pydantic>=2.0.0` is important).")
        print("5. **PyCharm Indentation:** If you see `IndentationError`, try pasting the code into a plain text editor first, then copy-paste into PyCharm, and run 'Reformat Code' (Ctrl+Alt+L or Cmd+Option+L).")
        print("6. **Error Messages:** Look closely at the error messages in the console. They often point directly to the problem.")
        print("7. **Ollama Model Response:** For debugging, examine the `--- DEBUG: SLM's Structured Research Decision RAW:` output. This is crucial! What is `should_research` and `search_query`? Is the JSON valid?")
        print("8. **Refinement:** If search results are found but the answer isn't improved, review the `refine_answer_with_research` prompt's instructions.")
        break