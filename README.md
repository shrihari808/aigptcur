# FRRUIT-AI Project

## Project Overview
FRRUIT-AI is a FastAPI-based AI backend service. The project is structured into two main services: `app_service` and `bg_service`.

## Project Structure

FRRUIT-AI/
├── app_service/
│   ├── api/
│   │   ├── __pycache__/
│   │   ├── fundamentals_rag/
│   │   ├── market_content/
│   │   ├── news_rag/
│   │   ├── youtube_rag/
│   │   ├── __init__.py
│   │   ├── caching.py
│   │   ├── chatbot.py
│   │   ├── reddit_chat.py
│   │   └── router.py

│   ├── csvdata/
│   ├── __init__.py
│   ├── config.py
│   └── main.py
└── bg_service/
    ├── api/
    │   ├── __init__.py
    │   ├── fundamentals_embedding.py
    │   ├── investor_stories.py
    │   ├── news_embeddings.py
    │   ├── reddit_embeddings.py
    │   ├── serp_api.py
    │   └── trending_stocks.py
    ├── csv_data/
    ├── __pycache__/
    ├── bg_app.py
    ├── config.py
    └── .env



## Setup and Installation

1. Clone the repository:
   ```
   git clone https://gitlab.com/jayranjeetbhatt/ai-gpt.git
   
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env` in the `bg_service` directory
   - Fill in the required environment variables

## Running the Services

### Running app_service

1. Navigate to the app_service directory:
   ```
   cd app_service
   ```

2. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`

3. Access the API documentation:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Running bg_service

1. Navigate to the bg_service directory:
   ```
   cd bg_service
   ```

2. Start the background service:
   ```
   uvicorn bg_app:app --reload
   ```
   This service will run on a different port, typically `http://localhost:8001`

## API Documentation
use the provided postman collection.

or

Both services provide automatic API documentation through Swagger UI and ReDoc.

- app_service:
  - Swagger UI: `http://localhost:8000/docs`
  - ReDoc: `http://localhost:8000/redoc`

- bg_service:
  - Swagger UI: `http://localhost:8001/docs`
  - ReDoc: `http://localhost:8001/redoc`

These interactive documentations provide detailed information about each endpoint, request/response models, and allow for testing the API directly from the browser.

## Service Descriptions

### 1. app_service

This service handles the main application logic and API endpoints.

- `api/`: Contains various modules for different functionalities.
  - `fundamentals_rag/`: Handles retrieval-augmented generation for financial fundamentals.
  - `market_content/`: includes documentchat and youtube summary.
  - `news_rag/`: Processes news content using retrieval-augmented generation.
  - `youtube_rag/`: Processes YouTube content using retrieval-augmented generation.
  - `caching.py`: Implements caching mechanisms for improved performance.
  - `chatbot.py`: Manages main chatbot functionality in a single api endpoint.
  - `reddit_chat.py`: Handles Reddit-specific chat features.
  - `graph_openai1.py`: Processes the user prompt and returns the appropriate data for plotting graphs.

  - `router.py`: Manages API routing using FastAPI.

- `csvdata/`: Stores CSV data files stock codes list and financial metrics.
- `config.py`: Contains configuration settings for the app service.
- `main.py`: The main entry point for the FastAPI app service.

### 2. bg_service

This service handles background tasks and data processing.

- `api/`: Contains modules for various data processing and API interactions.
  - `fundamentals_embedding.py`: Processes and creates or updates financial fundamentals data embeddings.
  - `investor_stories.py`: Generates  investor-stories content.
  - `news_embeddings.py`: Handles embeddings for news data.
  - `reddit_embeddings.py`: Processes Reddit data into embeddings.
  - `serp_api.py`: Interacts with search engine results pages (SERP) API.
  - `trending_stocks.py`: Analyzes or tracks trending stocks.
  - `autocomplete_news_questions.py`: generates autocomplete suggested prompts for news rag.
  - `suggestedprompts.py`: generates suggested prompts for fundamental data.
  

- `csv_data/`: Stores CSV data files stock codes list and financial metrics.
- `bg_app.py`: The main FastAPI application file for the background service.
- `config.py`: Configuration settings for the background service.
- `.env`: Environment variables file (keep this file secure)

