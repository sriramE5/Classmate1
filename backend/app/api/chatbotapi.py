from fastapi import APIRouter, Query, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
import markdown2
from fastapi.responses import JSONResponse
from typing import List
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import docx

# Load environment variables
load_dotenv()

# Setup Gemini Clients with two API keys
api_key1 = os.getenv("GEMINI_API_KEY1")
api_key2 = os.getenv("GEMINI_API_KEY2")

client1 = genai.Client(api_key=api_key1)
client2 = genai.Client(api_key=api_key2)

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str

class GoalItem(BaseModel):
    goal: str
    checked: bool

goals_data = []

# Store vectorstore and retriever in memory for demo (per session)
vectorstore = None
vector_index = None
rag_model = None

# Helper to extract text from DOCX
def extract_docx_text(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, vector_index, rag_model
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename[-5:]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    # Load and split document
    if file.filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    elif file.filename.lower().endswith(".docx"):
        text = extract_docx_text(tmp_path)
        # Wrap as LangChain Document
        from langchain.docstore.document import Document
        pages = [Document(page_content=text)]
    else:
        return JSONResponse({"error": "Unsupported file type. Only PDF and DOCX allowed."}, status_code=400)
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=['\n\n', '\n', ' '])
    chunks = splitter.split_documents(pages)
    # Embeddings and vectorstore
    api_key = api_key1 or api_key2
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    vector_index = vectorstore.as_retriever(search_kwargs={'k':8})
    # RAG model
    rag_model = RetrievalQA.from_chain_type(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key),
        retriever=vector_index,
        return_source_documents=True
    )
    return {"message": "File uploaded and processed successfully.", "chunks": len(chunks)}

class RAGRequest(BaseModel):
    query: str

@router.post("/api/rag_chat")
async def rag_chat(req: RAGRequest):
    global rag_model
    if rag_model is None:
        return JSONResponse({"error": "No document uploaded yet."}, status_code=400)
    try:
        response = rag_model({"query": req.query})
        answer = response["result"]
        # Optionally, include sources
        sources = [doc.metadata.get("source", "") for doc in response.get("source_documents", [])]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"error": str(e)}

async def generate_reply(prompt: str, client):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

@router.post("/api/chat")
async def chat(req: ChatRequest, as_markdown: bool = Query(False)):
    try:
        # First try with client1
        try:
            reply_text = await generate_reply(req.prompt, client1)
        except Exception as e1:
            # If error 503, switch to client2
            if "503" in str(e1) or "Service temporarily unavailable" in str(e1):
                try:
                    reply_text = await generate_reply(req.prompt, client2)
                except Exception as e2:
                    return {"reply": "Service is temporarily busy. Please try again later."}
            else:
                return {"reply": f"Error: {str(e1)}"}

        if as_markdown:
            reply_text = markdown2.markdown(reply_text)
        return {"reply": reply_text}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

@router.post("/api/goals")
async def save_goals(goals: list[GoalItem]):
    global goals_data
    goals_data = goals
    return {"message": "Goals saved successfully"}

@router.get("/api/goals")
async def get_goals():
    return goals_data

@router.get("/api/performance")
async def get_performance():
    total = len(goals_data)
    completed = sum(1 for g in goals_data if g.checked)
    percent = (completed / total) * 100 if total > 0 else 0
    return {
        "total": total,
        "completed": completed,
        "percent": percent
    }
