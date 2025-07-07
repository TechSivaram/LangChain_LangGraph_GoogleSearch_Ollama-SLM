# Smart Q&A System with Ollama SLM & Google Search

This project implements a smart Question & Answer (Q&A) system that leverages an Ollama-powered Small Language Model (SLM) for conversational AI and integrates with Google Search for real-time information retrieval. The system is designed to provide accurate, comprehensive, and up-to-date answers by deciding whether external web research is necessary. It is built with [LangChain](https://www.langchain.com/) for LLM orchestration and [LangGraph](https://langchain-ai.github.io/langgraph/) for robust agentic workflow management, and it features persistent chat history using SQLite, allowing for session management.

## Features

* **Intelligent Research Decision:** The SLM intelligently decides if a web search is needed to answer a user's query, especially for current or rapidly changing information.
* **Google Search Integration:** Utilizes Google Search API to fetch relevant and up-to-date information when research is required.
* **Answer Refinement:** The SLM refines its initial answer by synthesizing information obtained from web searches.
* **Conversation History Persistence:** Chat history is saved to a SQLite database, allowing users to continue conversations across sessions.
* **Flexible Ollama Model Selection:** Easily configure the desired Ollama model (e.g., `phi3`, `llama3`).
* **Error Handling and Troubleshooting:** Includes robust error handling and helpful troubleshooting tips for common issues.

## Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.9+:** Make sure you have a compatible Python version installed. [Download Python](https://www.python.org/downloads/)
2.  **Ollama:**
    * Download and install [Ollama](https://ollama.com/download).
    * Ensure the Ollama server is running. You can start it by running `ollama serve` in your terminal or by launching the desktop application.
    * Pull the desired language model. The default is `phi3`, but `llama3` is recommended for better tool-use capabilities. You can pull models using `ollama pull phi3` or `ollama pull llama3`.
3.  **Google Cloud Project and API Keys:**
    * **Google API Key:** Obtain a Google API Key from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials). Enable the **Custom Search API** for this key.
    * **Google Custom Search Engine ID (CSE ID):**
        * Go to the [Google Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/all) and create a new search engine.
        * Configure it to "Search the entire web."
        * Note down your Search Engine ID.
    * Make sure billing is enabled in your Google Cloud Project if you anticipate exceeding the free tier limits for the Custom Search API.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project uses various Python libraries, including [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), [python-dotenv](https://pypi.org/project/python-dotenv/) for environment variables, and [Pydantic](https://docs.pydantic.dev/latest/) for structured data. The full list is in `requirements.txt`:

    ```
    langchain>=0.2.5
    langchain-ollama>=0.0.3
    langgraph>=0.0.60
    langchain-community>=0.2.5
    langchain-google-community>=0.0.1
    langchain-core>=0.2.0
    google-search-results==2.0.0
    python-dotenv==1.0.1
    pydantic>=2.0.0
    pytz>=2024.1
    ```

    You can install them by running:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of your project and add your API keys and preferred Ollama model:
    ```dotenv
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    GOOGLE_CSE_ID="YOUR_GOOGLE_CSE_ID"
    OLLAMA_MODEL="phi3" # Or "llama3", "mistral", etc., ensure you have pulled it
    ```

    Replace `"YOUR_GOOGLE_API_KEY"` and `"YOUR_GOOGLE_CSE_ID"` with your actual keys.

## Usage

1.  **Run the application:**
    ```bash
    python main.py
    ```

2.  **Interact with the Q&A system:**
    * The system will first prompt you to start a `(n)ew` session or `(l)oad` an existing one.
    * Type your questions at the `Your Question:` prompt.
    * To end the current chat session and start a new one, type `new`.
    * To exit the application, type `exit`.

    Example interaction:
    ```
    --- Smart Q&A System with Ollama SLM & Google Search ---
    Using Ollama Model: 'phi3'
    Ensure Ollama is running and the specified model is pulled.
    Type 'exit' to quit.
    --- Database 'chat_history.db' initialized. ---

    Start a (n)ew session or (l)oad existing? (n/l): n
    --- New session started. Your session ID: f2a8e4c1-5d3b-4e0f-9a1b-2c7e6a0d4f1e ---

    [f2a8e4c1...] Your Question (type 'exit' to quit, 'new' to start new chat): What is the current population of India?

    --- Processing user question: 'What is the current population of India?' ---

    --- NODE: Calling SLM for Initial Answer ---
    --- DEBUG: SLM Initial Response: The population of India is a large and growing number, making it the most populous country in the world. As of recent estimates, it is over 1.4 billion people.

    --- NODE: Deciding if Research is Needed ---
    --- DEBUG: SLM's Structured Research Decision RAW: {"should_research": true, "search_query": "current population of India"} ---
    --- DEBUG: Research IS needed. Search Query: 'current population of India' ---

    --- NODE: Conducting Research ---

    --- DEBUG: NODE - Performing Google Search for query: 'current population of India' ---
    --- DEBUG: Search Results: Based on various sources, the current population of India is estimated to be over 1.4 billion people, making it the world's most populous country, surpassing China. The exact figure varies slightly depending on the source and the time of estimation, but it is generally accepted to be around 1,441,718,370 as of mid-2024. Sources include Worldometer, United Nations, and other demographic institutions.

    --- NODE: Refining Answer with Research Results ---
    --- DEBUG: SLM Refined Response: As of mid-2024, the current population of India is estimated to be over 1.4 billion people, specifically around 1,441,718,370. This makes the world's most populous country, having surpassed China.

    AI's Final Answer: As of mid-2024, the current population of India is estimated to be over 1.4 billion people, specifically around 1,441,718:370. This makes India the world's most populous country, having surpassed China.

    [f2a8e4c1...] Your Question (type 'exit' to quit, 'new' to start new chat): Who invented the lightbulb?

    --- Processing user question: 'Who invented the lightbulb?' ---

    --- NODE: Calling SLM for Initial Answer ---
    --- DEBUG: SLM Initial Response: The invention of the practical incandescent light bulb is often credited to Thomas Edison. However, many scientists and inventors contributed to the development of electric lighting.

    --- NODE: Deciding if Research is Needed ---
    --- DEBUG: SLM's Structured Research Decision RAW: {"should_research": false, "search_query": ""} ---
    --- DEBUG: SLM decided NO RESEARCH IS NEEDED. ---

    AI's Final Answer: The invention of the practical incandescent light bulb is most commonly attributed to Thomas Edison. However, it's important to note that many scientists and inventors contributed to the development of electric lighting over a period of time, with Edison's contribution being the commercially viable and long-lasting bulb.

    [f2a8e4c1...] Your Question (type 'exit' to quit, 'new' to start new chat): exit
    Exiting application.
    ```

## How It Works

The system uses a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine with the following nodes:

* **`call_slm_initial`**: The initial query is passed to the Ollama model to generate a preliminary response.
* **`decide_to_research`**: The SLM evaluates the user's question and the initial response to determine if external web research is required for accuracy, currency, or completeness. This node is explicitly prompted to be aggressive about using search for latest information.
* **`conduct_research`**: If research is deemed necessary, the `search_tool` (powered by Google Search API) is invoked with a generated query.
* **`refine_answer_with_research`**: The Ollama model receives the initial conversation history and the research results. It then synthesizes this information to provide a comprehensive, accurate, and up-to-date final answer.

The `decide_to_research` node uses a structured output ([Pydantic](https://docs.pydantic.dev/latest/) model) to reliably get the SLM's decision and search query.

## Project Structure

* `my_project/`
    * `├── .env`
    * `├── main.py`
    * `├── requirements.txt`
    * `└── chat_history.db` (generated at runtime)

## Troubleshooting

If you encounter issues, consider the following:

1.  **Ollama Server:** Ensure Ollama is running (`ollama serve` or the desktop app).
2.  **Ollama Model:** Verify that the configured `OLLAMA_MODEL` (e.g., `phi3`, `llama3`) is pulled locally (`ollama list`). `llama3` is generally more reliable for tool use.
3.  **Google API Keys:**
    * Double-check `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in your `.env` file.
    * Confirm that the Custom Search API is enabled in your Google Cloud Project.
    * Ensure your Programmable Search Engine is configured to search the entire web.
    * Check Google Cloud billing if you suspect exceeding free tier limits.
4.  **Dependencies:** Make sure all dependencies from `requirements.txt` are installed in your environment. Pay close attention to `pydantic>=2.0.0` and `langchain-core>=0.2.0`.
5.  **Indentation Errors:** If you see `IndentationError` (common after copy-pasting code), try pasting the code into a plain text editor first, then copy-paste into your IDE (like PyCharm), and run its 'Reformat Code' function (e.g., Ctrl+Alt+L or Cmd+Option+L).
6.  **Error Messages:** Look closely at the full error messages in the console. They often provide direct clues about the problem.
7.  **SLM Decision Debugging:** Examine the `--- DEBUG: SLM's Structured Research Decision RAW:` output. This shows what the SLM decided regarding research and the generated search query. This is crucial for understanding why research might or might not be happening.
8.  **Refinement Quality:** If search results are found but the answer isn't significantly improved, review the `refine_answer_with_research` prompt in `main.py` to ensure it guides the SLM effectively.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is open-source and available under the MIT License.