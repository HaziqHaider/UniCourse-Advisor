# IntelliCourse - AI-Powered University Course Advisor

IntelliCourse is an intelligent course advisory system that helps students discover and learn about university courses using AI. The system combines Retrieval-Augmented Generation (RAG) with web search capabilities to provide comprehensive answers to course-related queries.

## Features

- **Smart Query Routing**: Automatically classifies questions as course-related or general knowledge
- **Course Catalog Search**: Vector-based search through university course catalogs (PDF documents)
- **Web Integration**: Augments responses with current information from web search
- **AI-Powered Responses**: Uses Google Gemini AI to generate natural, helpful answers
- **REST API**: FastAPI-based endpoint for easy integration

Working Guide:
First create a data folder in intelicourse and then create a pdf folder and then add the pdfs you want.
cd intellicourse
pip install -r requirements.txt
Set Up Environment Variables:
Create a .env file in the project root:
env:
GOOGLE_API_KEY=your_google_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
Add Course Catalog PDFs:
Place your university course catalog PDF files in the data/pdfs/ directory:
Start Running Application:
 uvicorn main:app --reload --port 8000
 URL: http://127.0.0.1:8000/chat

