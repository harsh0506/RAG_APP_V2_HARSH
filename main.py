#### This is FASTAPI service created to work with rag class. Please Implment the rag class before using it. Below are the steps happing in the code.
#### Create a templates folder in root and put index.html and upload.html file in it.

# The application sets up a FastAPI server with WebSocket support
# It uses Jinja2 templates for HTML rendering
# A RAGApplication class is instantiated as 'rag_app'
# Sessions are managed using a dictionary
# The app provides endpoints for:

# Uploading PDF files
# Initiating chat sessions
# Handling chat requests via HTTP POST and WebSocket


## The upload process:

# Validates the file is a PDF
# Generates a unique session ID
# Saves the file and initializes the RAG system


## The chat process:

# Uses the RAG system to generate answers
# Maintains chat history per session
# Provides processing time and sources information


# WebSocket endpoint allows for real-time chat interaction
# Error handling is implemented for various scenarios
# The app can be run using Uvicorn server

from fastapi import FastAPI, WebSocket, WebSocketDisconnect,HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

import shutil
from pathlib import Path
import time
import uuid
import os
import pathlib

from pydantic import BaseModel
from rag import RAGApplication

app = FastAPI()

templates = Jinja2Templates(directory="templates")

rag_app = RAGApplication()

sessions = {}

class ChatRequest(BaseModel):
    question: str
    chat_history: list

class UploadResponse(BaseModel):
    session_id: str
    
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.get("/", response_class=HTMLResponse)
async def get_upload_page():
    template_dir = 'templates'
    template_file = 'upload.html'
    template_path = os.path.join(template_dir, template_file)
    content = Path(template_path).read_text()
    return HTMLResponse(content=content, status_code=200)

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    session_id = str(uuid.uuid4())
    file_location = pathlib.Path("documents") / file.filename
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        documents = rag_app.load_documents(file_location)
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to load document")
        
        rag_app.initialize(documents, Path(file_location).name)
        sessions[session_id] = {"file_path": file_location, "chat_history": []}
        
        # Redirect to the chat page
        return RedirectResponse(url=f"/chat/{session_id}", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def get_chat_page(session_id: str):
    get_session(session_id)  # Validate session exists
    template_dir = 'templates'
    template_file = 'index.html'
    template_path = os.path.join(template_dir, template_file)
    content = Path(template_path).read_text()
    return HTMLResponse(content=content)

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: ChatRequest):
    session = get_session(session_id)
    
    start_time = time.time()
    try:
        answer, sources = await rag_app.achat(request.question, session["chat_history"])
        session["chat_history"].append((request.question, answer))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    
    processing_time = time.time() - start_time
    return {
        "processing_time": processing_time,
        "answer": answer,
        "sources": sources
    }
    
@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        session = get_session(session_id)
        await websocket.accept()
        
        while True:
            data = await websocket.receive_text()
            if data.lower() == 'quit':
                await websocket.send_json({"type": "info", "message": "Chat ended."})
                break
            
            start_time = time.time()
            try:
                answer, sources = await rag_app.achat(data, session["chat_history"])
                session["chat_history"].append((data, answer))
                processing_time = time.time() - start_time

                await websocket.send_json({
                    "type": "answer",
                    "processing_time": processing_time,
                    "answer": answer,
                    "sources": "sources"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"An error occurred while processing your request: {str(e)}"
                })
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"An error occurred: {str(e)}"
        })
    finally:
        print(f"WebSocket connection closed for session {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
