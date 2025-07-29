"""
Unified Chatbot & Advanced RAG API Router for FastAPI
- Advanced RAG: persistent, multi-file, per-user, robust
- Regular chat: /api/chat
"""
import os
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.api.userapi import get_current_user
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import docx
from langchain_community.document_loaders import PyPDFLoader
import markdown2

# Load environment variables
load_dotenv()
api_key1 = os.getenv("GEMINI_API_KEY1")
api_key2 = os.getenv("GEMINI_API_KEY2")
api_key = api_key1 or api_key2

router = APIRouter()

UPLOAD_ROOT = "./uploads"
CHROMA_ROOT = "./chroma_db"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
K_RETRIEVAL = 12

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbotapi")

class RAGRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    prompt: str

# Helper: Save uploaded file
def save_user_file(user_id: str, file: UploadFile) -> str:
    user_dir = os.path.join(UPLOAD_ROOT, user_id)
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return file_path

# Helper: Extract text from DOCX
def extract_docx_text(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Helper: Get Chroma vectorstore for user
def get_user_vectorstore(user_id: str, embeddings=None) -> Chroma:
    persist_directory = os.path.join(CHROMA_ROOT, user_id)
    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

# Helper: Index a file (PDF or DOCX) for a user
def index_file(user_id: str, file_path: str, file_name: str, upload_time: str, embeddings=None) -> int:
    # Load and split document
    if file_name.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    elif file_name.lower().endswith(".docx"):
        text = extract_docx_text(file_path)
        from langchain.docstore.document import Document
        pages = [Document(page_content=text)]
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX allowed.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '.', ' ']
    )
    chunks = splitter.split_documents(pages)
    # Add metadata to each chunk
    for idx, chunk in enumerate(chunks):
        chunk.metadata["file_name"] = file_name
        chunk.metadata["upload_time"] = upload_time
        chunk.metadata["chunk_index"] = idx
    # Add to persistent vectorstore
    vectorstore = get_user_vectorstore(user_id, embeddings)
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    logger.info(f"Indexed {len(chunks)} chunks for user {user_id}, file {file_name}")
    return len(chunks)

# Helper: List uploaded files for a user
def list_user_files(user_id: str) -> List[Dict[str, Any]]:
    user_dir = os.path.join(UPLOAD_ROOT, user_id)
    if not os.path.exists(user_dir):
        return []
    files = []
    for fname in os.listdir(user_dir):
        fpath = os.path.join(user_dir, fname)
        if os.path.isfile(fpath):
            files.append({
                "file_name": fname,
                "size": os.path.getsize(fpath),
                "upload_time": datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat()
            })
    return files

# Helper: Delete a file and its chunks for a user
def delete_user_file(user_id: str, file_name: str):
    # Delete file from uploads
    user_dir = os.path.join(UPLOAD_ROOT, user_id)
    file_path = os.path.join(user_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    # Remove chunks from vectorstore
    persist_directory = os.path.join(CHROMA_ROOT, user_id)
    if not os.path.exists(persist_directory):
        return
    # Rebuild vectorstore without the deleted file's chunks
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    # Get all docs, filter out those from the deleted file
    all_docs = vectorstore.get()['documents']
    all_metas = vectorstore.get()['metadatas']
    keep_docs = []
    keep_metas = []
    for doc, meta in zip(all_docs, all_metas):
        if meta.get("file_name") != file_name:
            keep_docs.append(doc)
            keep_metas.append(meta)
    # Clear and re-add
    vectorstore.delete_collection()
    if keep_docs:
        vectorstore.add_texts(keep_docs, metadatas=keep_metas)
        vectorstore.persist()
    logger.info(f"Deleted file {file_name} and its chunks for user {user_id}")

# Endpoint: Upload and index a file (advanced, persistent, multi-file, per-user)
@router.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    # Save file
    file_path = save_user_file(user_id, file)
    upload_time = datetime.utcnow().isoformat()
    # Index file
    try:
        n_chunks = index_file(user_id, file_path, file.filename, upload_time)
        return {"message": "File uploaded and indexed", "chunks": n_chunks, "file": file.filename}
    except Exception as e:
        logger.error(f"Upload/index error for user {user_id}, file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index file: {e}")

# Endpoint: List uploaded files
@router.get("/api/files")
async def files(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    files = list_user_files(user_id)
    return {"files": files}

# Endpoint: Delete a file and its chunks
@router.delete("/api/delete_file")
async def delete_file(
    file_name: str = Query(...),
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    try:
        delete_user_file(user_id, file_name)
        return {"message": f"Deleted {file_name}"}
    except Exception as e:
        logger.error(f"Delete error for user {user_id}, file {file_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

# Endpoint: Advanced RAG chat (persistent, multi-file, per-user)
@router.post("/api/rag_chat")
async def rag_chat(
    req: RAGRequest,
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    vectorstore = get_user_vectorstore(user_id, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": K_RETRIEVAL})
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key),
        retriever=retriever,
        return_source_documents=True
    )
    try:
        response = rag_chain({"query": req.query})
        answer = response["result"]
        sources = [
            {
                "file": doc.metadata.get("file_name", ""),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "upload_time": doc.metadata.get("upload_time", ""),
                "preview": doc.page_content[:200]
            }
            for doc in response.get("source_documents", [])
        ]
        logger.info(f"RAG query for user {user_id}: '{req.query}' | Answer: {answer[:80]}...")
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"RAG error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

# Endpoint: Regular chat (non-RAG)
@router.post("api/chat")  # Regular chat endpoint
async def chat(req: ChatRequest):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
        response = llm.invoke(req.prompt)
        return {"reply": response}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}



# Helper: Generate reply for /api/chat
async def generate_reply(prompt: str, client):
    response = client.invoke(prompt)
    return response
