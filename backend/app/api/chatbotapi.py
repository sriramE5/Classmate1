from fastapi import APIRouter,FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
import json

# Load environment variables
load_dotenv()

# Setup Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Initialize FastAPI
router = APIRouter()

# Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# In-memory storage (replace with DB or file in production)
goals_data = []

class ChatRequest(BaseModel):
    prompt: str

class GoalItem(BaseModel):
    goal: str
    checked: bool

@router.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=req.prompt
        )
        return {"reply": response.text}
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
import markdown2

@router.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=req.prompt
        )
        markdown_html = markdown2.markdown(response.text)
        return {"reply": markdown_html}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}