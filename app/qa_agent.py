import os
from dotenv import load_dotenv
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing import List, Literal, TypedDict, Annotated
import operator


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

# Debugging print to confirm OLLAMA_MODEL is loaded
print(f"QA_AGENT.PY: OLLAMA_MODEL configured as: '{OLLAMA_MODEL}'")


# Initialize components
llm = ChatOllama(model=OLLAMA_MODEL)
search_tool = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

# Define Pydantic models for structured output
class ResearchDecision(BaseModel):
    answer: str = Field(description="The initial answer to the user's question based on existing knowledge.")
    should_research: bool = Field(description="Whether a web search is needed.")
    search_query: str = Field(description="The search query to use if research is needed.")

# LangGraph state
class AgentState(TypedDict):
    chat_history: Annotated[List[BaseMessage], operator.add]
    research_results: str
    user_question: str
    should_research: bool
    search_query: str
    refine_answer_with_research: str # To carry initial answer


# --- Node Functions ---

def call_slm_initial(state: AgentState, llm_model: ChatOllama):
    """
    Initial call to LLM to get an answer and decide if research is needed.
    Includes a programmatic override for critical factual queries.
    """
    print("--- NODE: Calling SLM for Initial Answer ---")
    question = state["user_question"]
    chat_history = state["chat_history"]

    # --- NEW PROGRAMMATIC OVERRIDE FOR CRITICAL QUERIES ---
    force_research = False
    forced_search_query = ""
    lower_question = question.lower()

    # Keywords that trigger a forced search (e.g., for current political figures)
    critical_keywords = ["chief minister", "cm", "president", "prime minister", "governor", "current leader", "latest leader", "who is"]
    
    # Check if any critical keyword is in the question
    if any(keyword in lower_question for keyword in critical_keywords):
        force_research = True
        # Construct a more specific search query for political leaders
        if "chief minister" in lower_question or "cm" in lower_question:
            # Attempt to extract the state/country from the question
            # Simple heuristic: remove common leading phrases and trim
            location_part = lower_question.replace("who is", "").replace("chief minister of", "").replace("cm of", "").strip()
            forced_search_query = f"current chief minister of {location_part}"
        elif "president" in lower_question:
            location_part = lower_question.replace("who is", "").replace("president of", "").strip()
            forced_search_query = f"current president of {location_part}"
        elif "prime minister" in lower_question:
            location_part = lower_question.replace("who is", "").replace("prime minister of", "").strip()
            forced_search_query = f"current prime minister of {location_part}"
        else: # Generic current leader query if specific role not found but keyword exists
             location_part = lower_question.replace("who is", "").replace("current leader of", "").replace("latest leader of", "").strip()
             forced_search_query = f"current leader of {location_part}"
        
        # Add a default if extraction fails or is too broad
        if not forced_search_query:
            forced_search_query = question + " current" # Fallback to original question + " current"

        print(f"--- DEBUG: Programmatically forcing research for: '{question}' with query: '{forced_search_query}' ---")
    # --- END NEW PROGRAMMATIC OVERRIDE ---

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
        # The prompt below is for the LLM's own decision-making,
        # but the programmatic override will take precedence if triggered.
        ("user", "Based on the conversation history and your knowledge, provide an initial concise answer. **Crucially, if the question involves current political leaders (like 'Chief Minister', 'President', 'Prime Minister'), recent events, or information that changes frequently, you MUST perform a web search to ensure accuracy and currency.** In such cases, set 'should_research' to true and provide a precise search query. Otherwise, if you are absolutely confident your internal knowledge is sufficient and up-to-date for a stable fact, set 'should_research' to false. Your output must be a JSON object with 'answer', 'should_research' (boolean), and 'search_query' (string, if research is needed, otherwise empty).")
    ])

    structured_llm = llm_model.with_structured_output(ResearchDecision)
    chain = prompt | structured_llm

    response = chain.invoke({"question": question, "chat_history": chat_history})
    
    # Use the LLM's response, but override should_research and search_query if force_research is true
    response_answer = response.answer
    response_should_research = response.should_research
    response_search_query = response.search_query

    if force_research:
        response_should_research = True
        # If the LLM also provided a valid-looking query, consider merging or preferring ours.
        # For simplicity, if we forced it, we use our derived query.
        if forced_search_query: # Ensure our forced query is not empty
             response_search_query = forced_search_query
        elif not response_search_query: # If LLM gave no query but we forced search
             response_search_query = question # Fallback to original question as search query
        
        print(f"--- DEBUG: OVERRIDE APPLIED. should_research={response_should_research}, search_query='{response_search_query}' ---")
    
    print(f"--- DEBUG: SLM Initial Response (Answer): {response_answer}")
    print(f"--- DEBUG: SLM's Structured Research Decision RAW: {{\"answer\":\"{response_answer}\",\"should_research\":{response_should_research},\"search_query\":\"{response_search_query}\"}}")

    return {
        "chat_history": chat_history + [HumanMessage(content=question), AIMessage(content=response_answer)],
        "should_research": response_should_research,
        "search_query": response_search_query,
        "refine_answer_with_research": response_answer
    }

def decide_to_research(state: AgentState, llm_model: ChatOllama, decision_model: BaseModel):
    """
    Decides whether to proceed with research based on the LLM's initial output (or override).
    """
    print("--- NODE: Deciding if Research is Needed ---")
    current_should_research = state["should_research"] # Get the value from the state passed from previous node
    search_query = state["search_query"]

    # --- NEW DEBUG PRINTS HERE ---
    print(f"--- DEBUG: Inside decide_to_research. State's should_research value: {current_should_research}, search_query: '{search_query}' ---")
    # --- END NEW DEBUG PRINTS ---

    if current_should_research: # Use the value from the state
        print(f"--- DEBUG: decide_to_research: Branching to conduct_research. ---")
        return {"should_research": True, "search_query": search_query}
    else:
        print(f"--- DEBUG: decide_to_research: Branching to refine_answer_with_research directly (no research). ---")
        return {"should_research": False, "search_query": ""}

def conduct_research(state: AgentState, search_wrapper: GoogleSearchAPIWrapper):
    """
    Conducts web research using Google Search API if a query is provided and needed.
    """
    print("--- NODE: Conducting Research ---")
    search_query = state["search_query"]
    if not search_query or not state["should_research"]:
        print("--- DEBUG: No search query or research not needed. Skipping search. ---")
        return {"research_results": ""}

    print(f"--- DEBUG: NODE - Performing Google Search for query: '{search_query}' ---")
    try:
        search_results = search_wrapper.run(search_query)
        print(f"--- DEBUG: Search Results: {search_results}")
        return {"research_results": search_results}
    except Exception as e:
        print(f"--- ERROR: Google Search failed: {e} ---")
        return {"research_results": f"Failed to conduct search: {e}"}


def refine_answer_with_research(state: AgentState, llm_model: ChatOllama):
    """
    Refines the initial answer using research results, if available.
    """
    print("--- NODE: Refining Answer with Research Results ---")
    user_question = state["user_question"]
    chat_history = state["chat_history"]
    research_results = state["research_results"]
    initial_answer = state["refine_answer_with_research"]

    if research_results and "Failed to conduct search" not in research_results and research_results.strip() != "":
        prompt_messages = [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", f"Here is the original question: {user_question}"),
            ("user", f"I initially considered this answer: {initial_answer}"),
            ("user", f"**Here are highly relevant and up-to-date web search results that MUST be used to formulate the final answer:**\n\n{research_results}\n\n"
                     "**Based SOLELY on the provided web search results, and ignoring any prior knowledge that conflicts, what is the most accurate and current answer to the original question?** "
                     "Your answer must be concise, directly answer the question, and be supported by the search results. "
                     "If the search results explicitly state a current fact that contradicts the initial answer, you MUST use the information from the search results. "
                     "Do not mention knowledge cutoffs or suggest consulting other sources if the answer is clearly present in the provided search results.")
        ]
    else:
        print("--- DEBUG: No valid research results for refinement. Using initial answer. ---")
        # If no research was conducted or failed, use the initial answer from SLM
        # The chat_history already contains the user question and the initial AI response.
        # We ensure the final state carries this initial answer.
        return {"refine_answer_with_research": initial_answer}


    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt | llm_model

    response_message = chain.invoke({"user_question": user_question, "chat_history": chat_history, "research_results": research_results})
    refined_answer = response_message.content
    print(f"--- DEBUG: SLM Refined Response: {refined_answer}")

    updated_chat_history_for_state = chat_history + [AIMessage(content=refined_answer)]
    return {"chat_history": updated_chat_history_for_state, "refine_answer_with_research": refined_answer}


# --- Function to create and return the LangGraph agent ---
def get_qa_agent():
    print("--- QA_AGENT.PY: Entering get_qa_agent function to build workflow. ---")
    workflow = StateGraph(AgentState)

    workflow.add_node("call_slm_initial", lambda state: call_slm_initial(state, llm))
    workflow.add_node("decide_to_research", lambda state: decide_to_research(state, llm, ResearchDecision))
    workflow.add_node("conduct_research", lambda state: conduct_research(state, search_tool))
    workflow.add_node("refine_answer_with_research", lambda state: refine_answer_with_research(state, llm))

    workflow.set_entry_point("call_slm_initial")
    # --- THIS IS THE CRUCIAL MISSING LINE ADDED TO CONNECT THE ENTRY POINT! ---
    workflow.add_edge("call_slm_initial", "decide_to_research")
    # -------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "decide_to_research",
        lambda state: "conduct_research" if state["should_research"] else "refine_answer_with_research",
        {"conduct_research": "conduct_research", "refine_answer_with_research": "refine_answer_with_research"}
    )
    workflow.add_edge("conduct_research", "refine_answer_with_research")
    workflow.add_edge("refine_answer_with_research", END)

    app = workflow.compile()
    print("--- QA_AGENT.PY: Workflow compiled. ---")
    return app

# Example usage (for testing qa_agent.py directly, optional)
if __name__ == "__main__":
    print("--- QA_AGENT.PY: Running direct test of agent initialization. ---")