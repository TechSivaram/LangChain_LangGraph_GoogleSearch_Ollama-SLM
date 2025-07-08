# LangChain_LangGraph_GoogleSearch_Ollama-llama3

## 💡 Project Overview

This project, `LangChain_LangGraph_GoogleSearch_Ollama-llama3`, implements a robust and intelligent Question-Answering (QA) system. It leverages the power of Large Language Models (LLMs) run locally via Ollama, combined with real-time web search capabilities through the Google Custom Search API. The system intelligently decides when to use its internal knowledge versus performing a targeted web search, ensuring up-to-date and accurate answers, especially for dynamic or current events.

### ✨ Key Features

* **Intelligent QA:** Provides answers to user questions using a sophisticated LLM (`llama3` by default).
* **Real-time Web Search Integration:** Automatically performs Google searches for queries requiring the latest information (e.g., current events, political figures).
* **Programmatic Search Override:** Includes logic to ensure critical, time-sensitive queries always trigger a web search for maximum accuracy, overriding the LLM's initial decision if necessary.
* **Modular Agentic Architecture:** Built with LangChain and LangGraph for a clear, extensible, and stateful multi-step workflow.
* **Local LLM Execution:** Utilizes Ollama to run the LLM locally on your machine, ensuring data privacy and reducing reliance on external API services for model inference.
* **User-friendly Web Interface:** Accessible via a simple web UI provided by a FastAPI backend.

### 🎯 Problem Solved

Large Language Models (LLMs) often have a "knowledge cutoff," meaning their training data is not current and they cannot provide up-to-the-minute information. This project addresses this limitation by dynamically integrating live web search. For questions where timeliness is crucial, `LangChain_LangGraph_GoogleSearch_Ollama-llama3` ensures answers are always fresh and reliable, making it a powerful tool for information retrieval on dynamic topics.

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **LangChain:** For building LLM applications and orchestrating components.
* **LangGraph:** A library for building robust and stateful multi-step LLM agents and workflows.
* **FastAPI:** A modern, fast (high-performance) web framework for building the API backend.
* **Uvicorn:** ASGI server for running the FastAPI application.
* **Ollama:** For easily running large language models (like Llama 3) locally on your machine.
* **Google Search API:** For performing real-time web searches to gather up-to-date information.
* **`python-dotenv`:** For securely managing environment variables (API keys, model names, etc.).
* **`pydantic`:** For data validation and settings management (used by FastAPI/LangChain).

## 🚀 Getting Started

Follow these steps to set up and run `LangChain_LangGraph_GoogleSearch_Ollama-llama3` on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.9+**
* **pip** (Python package installer)
* **Ollama:** Download and install the Ollama desktop application or server from [ollama.com](https://ollama.com/).
* **Google Cloud Project & API Key:** You'll need a Google Search API Key and a Custom Search Engine ID (CSE ID) enabled for the Custom Search JSON API. Follow [Google's official guide](https://developers.google.com/custom-search/v1/overview) to get these credentials.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LangChain_LangGraph_GoogleSearch_Ollama-llama3.git # Replace with your actual repo URL
    cd LangChain_LangGraph_GoogleSearch_Ollama-llama3
    ```

2.  **Create a Python virtual environment (highly recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Make sure you have a `requirements.txt` file in your project's root containing: `fastapi`, `uvicorn`, `langchain`, `langchain-ollama`, `langchain-google-community`, `langgraph`, `python-dotenv`, `pydantic`).

### Configuration

1.  **Create a `.env` file:**
    In the root of your project directory, create a file named `.env` and add your API keys and model configuration. This file should **not** be committed to public repositories for security reasons (it's typically listed in `.gitignore`).
    ```dotenv
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    GOOGLE_CSE_ID="YOUR_GOOGLE_CSE_ID"
    OLLAMA_MODEL="llama3" # This is the default. You can change to "phi3" or another model you've pulled.
    ```
    **Replace `"YOUR_GOOGLE_API_KEY"` and `"YOUR_GOOGLE_CSE_ID"` with your actual credentials.**

2.  **Pull the Ollama Model:**
    Open a *separate* terminal or command prompt window and ensure your Ollama server is running. Then, download the `llama3` model (or your chosen model):
    ```bash
    ollama pull llama3
    ```
    To ensure the Ollama server is running in the background:
    ```bash
    ollama serve
    ```
    Keep this terminal window open while your FastAPI application is running.

### Running the Application

1.  **Start the FastAPI application:**
    Open a terminal in your project's root directory (with the Python virtual environment activated) and run:
    ```bash
    uvicorn main:app --reload
    ```
    The `--reload` flag is useful during development as it automatically restarts the server when code changes are detected.

2.  **Access the Application:**
    Open your web browser and navigate to `http://127.0.0.1:8000/`. You should see the user interface ready for interaction.

## 🖥️ Frontend UI

The user interface for this project is a simple, browser-based chat interface. It is implemented using basic HTML, CSS, and JavaScript, and these static files are served directly by the FastAPI backend (`main.py`). This design provides a lightweight and accessible way to interact with the QA system without requiring a separate frontend development server or complex build process.

### UI File Location:

The primary UI file (`index.html`) and any associated static assets (like `style.css` or `script.js`) are located within the `static/` directory in the project's root. FastAPI is configured to serve these files automatically.

### UI Features:

* **Input Field:** A text box for typing your questions.
* **Send Button:** To submit your query to the backend.
* **Conversation History:** Displays previous questions and the system's responses, forming a simple chat log.

This minimalist UI focuses on functionality, allowing you to quickly test and interact with the LLM and search capabilities.

## 💡 Usage

Once the application is running, you can interact with the system by typing your questions into the chat interface.

**Examples of queries you can try:**

* "What is the capital of France?" (May use LLM's internal knowledge)
* "Who is the current chief minister of Andhra Pradesh?" (Designed to trigger a web search due to programmatic override and timeliness)
* "What is the weather like in Vizianagaram today?" (Likely triggers a web search for real-time data)
* "Explain the concept of quantum entanglement." (Primarily uses LLM's internal knowledge, possibly augmented by general search if model deems necessary)

The underlying LangGraph agent will intelligently decide whether to answer based on the LLM's knowledge, perform a targeted web search, or combine both to provide the most accurate and up-to-date response.

## 📂 Project Structure

The project's directory and file organization is designed for clarity and maintainability:

* `.env`: Stores environment-specific variables like API keys (`GOOGLE_API_KEY`, `GOOGLE_CSE_ID`) and model configurations (`OLLAMA_MODEL`). **This file is crucial for security and is intentionally excluded from version control via `.gitignore`.**
* `main.py`: Serves as the primary entry point for the FastAPI application, handling API route definitions and serving the static web user interface.
* `app/`: A Python package encapsulating the core business logic of the application.
    * `__init__.py`: Designates `app` as a Python package.
    * `qa_agent.py`: Contains the intricate LangGraph agent's state machine, defining how the LLM interacts with tools (like Google Search), manages conversational state, and applies programmatic overrides for specific queries.
* `static/`: Dedicated to serving all static frontend assets directly to the user's browser.
    * `index.html`: The main interactive web interface for the chat application.
    * *(Optional: `style.css`, `script.js`)*: Additional CSS for styling and JavaScript for client-side interactivity, if present.
* `requirements.txt`: Lists all Python package dependencies required for the project.
* `LICENSE`: Specifies the project's legal licensing terms, defining how others can use, distribute, and modify the code.
* `README.md`: This comprehensive documentation file, providing an overview, setup guide, usage instructions, and more.

## ⚠️ Troubleshooting

* **"Failed to conduct search" / No search results:**
    * Ensure your `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are correctly set in your `.env` file.
    * Verify that the Google Custom Search JSON API is enabled for your Google Cloud project linked to the API key.
    * Check your internet connection.
* **Slow responses from Ollama model (e.g., Llama 3):**
    * **GPU Usage:** This is the most common reason for slowness. Ensure your system has a compatible GPU (NVIDIA with CUDA, AMD with ROCm) and sufficient VRAM (8GB+ for `llama3:8b` is highly recommended, 16GB+ is ideal). Ollama should automatically utilize your GPU if available and properly configured. Use `nvidia-smi` (for NVIDIA GPUs on Linux/WSL) or your system's task/activity monitor to check GPU utilization by Ollama.
    * **System RAM:** Ensure you have enough free system RAM (16GB+ recommended). If the model doesn't fit entirely into VRAM, it will use system RAM, which is significantly slower.
    * **Ollama Version:** Keep your Ollama installation updated (`ollama --version`, then `ollama update` or re-download from [ollama.com](https://ollama.com/)). Newer versions often include performance optimizations.
    * **CPU Threads:** You can experiment with setting the `OLLAMA_NUM_THREADS` environment variable to your CPU's physical core count before starting Ollama or Uvicorn. For example, `export OLLAMA_NUM_THREADS=8` (on Linux/macOS) or `set OLLAMA_NUM_THREADS=8` (on Windows CMD).
* **"Connection Refused" to Ollama:**
    * Make sure `ollama serve` is running in a *separate* terminal window and that the Ollama application is active on your system.
* **Incorrect answers despite seemingly good search results:**
    * If the search results clearly contain the correct answer but the LLM provides an incorrect one, it might be an issue with the LLM's ability to extract and synthesize information from the provided context. Ensure the prompts within `qa_agent.py` for tasks like `refine_answer_with_research` are clear and specific.

## 🗺️ Roadmap (Future Enhancements)

* Implement proper conversational memory to allow for multi-turn conversations and context retention.
* Explore more sophisticated RAG (Retrieval-Augmented Generation) techniques, such as document chunking and embedding, for internal knowledge bases.
* Enhance error handling and provide more informative user feedback in the web UI.
* Add Dockerfile and `docker-compose.yml` for easier containerized deployment.
* Allow users to select different locally-pulled Ollama models directly from the UI.
* Integrate alternative web search providers beyond Google Search API.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/LangChain_LangGraph_GoogleSearch_Ollama-llama3/issues) (replace with your actual issues page link).

1.  **Fork** the repository.
2.  **Create a new branch** for your feature or bugfix: `git checkout -b feature/your-feature-name`
3.  **Make your changes**.
4.  **Commit your changes:** `git commit -m "feat: Add new feature X"`
5.  **Push to your branch:** `git push origin feature/your-feature-name`
6.  **Open a Pull Request** to the `main` branch of this repository.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details. (If you don't have one, consider adding a `LICENSE` file in your root directory).

## 📧 Contact

For any questions, feedback, or collaborations, please reach out to [Your Name/Email or GitHub Profile Link].

## 🖼️ Gallery

<img width="1276" alt="image" src="https://github.com/user-attachments/assets/be05ceb0-0086-402c-bc49-91f52a501cfd" />

<img width="1280" alt="image" src="https://github.com/user-attachments/assets/a86d3e3c-17c8-4858-b5c0-1174a389424a" />

<img width="1277" alt="image" src="https://github.com/user-attachments/assets/8877bb8f-40ca-4e8c-affc-80d374caec8c" />

<img width="1277" alt="image" src="https://github.com/user-attachments/assets/bf19da9b-c5f4-490e-9f06-17d9ea2a93d6" />

