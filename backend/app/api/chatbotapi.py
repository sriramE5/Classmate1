from fastapi import APIRouter, Query, Depends, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
import markdown2
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import certifi
from jose import jwt, JWTError
from bson import ObjectId
from backend.app.api.userapi import get_current_user

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Setup Gemini Clients with two API keys
api_key1 = os.getenv("GEMINI_API_KEY1")
api_key2 = os.getenv("GEMINI_API_KEY2")

client1 = genai.Client(api_key=api_key1)
client2 = genai.Client(api_key=api_key2)

# MongoDB connection
try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tlsCAFile=certifi.where()
    )
    client.server_info()
    db = client["classmate"]
    goals_collection = db["goals"]
    notes_collection = db["notes"]
    tasks_collection = db["tasks"]
except ConnectionFailure as e:
    db = None
    goals_collection = None
    notes_collection = None
    tasks_collection = None

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str

class GoalItem(BaseModel):
    goal: str
    checked: bool

class NoteItem(BaseModel):
    content: str
    timestamp: str  # or use datetime if you want

class TaskItem(BaseModel):
    task: str
    checked: bool

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
async def save_goals(goals: list[GoalItem], current_user: dict = Depends(get_current_user)):
    if goals_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    user_id = str(current_user["_id"])
    # Remove existing goals for user
    goals_collection.delete_many({"user_id": user_id})
    # Insert new goals
    for goal in goals:
        goals_collection.insert_one({
            "user_id": user_id,
            "goal": goal.goal,
            "checked": goal.checked
        })
    return {"message": "Goals saved successfully"}

@router.get("/api/goals")
async def get_goals(current_user: dict = Depends(get_current_user)):
    if goals_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    user_id = str(current_user["_id"])
    user_goals = list(goals_collection.find({"user_id": user_id}))
    return [
        {"goal": g["goal"], "checked": g["checked"]}
        for g in user_goals
    ]

@router.get("/api/performance")
async def get_performance(current_user: dict = Depends(get_current_user)):
    if goals_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    user_id = str(current_user["_id"])
    user_goals = list(goals_collection.find({"user_id": user_id}))
    total = len(user_goals)
    completed = sum(1 for g in user_goals if g["checked"])
    percent = (completed / total) * 100 if total > 0 else 0
    return {
        "total": total,
        "completed": completed,
        "percent": percent
    }

@router.post("/api/notes")
async def save_notes(notes: list[NoteItem], current_user: dict = Depends(get_current_user)):
    if notes_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    user_id = str(current_user["_id"])
    notes_collection.delete_many({"user_id": user_id})
    for note in notes:
        notes_collection.insert_one({
            "user_id": user_id,
            "content": note.content,
            "timestamp": note.timestamp
        })
    return {"message": "Notes saved successfully"}

@router.get("/api/notes")
async def get_notes(current_user: dict = Depends(get_current_user)):
    if notes_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    user_id = str(current_user["_id"])
    user_notes = list(notes_collection.find({"user_id": user_id}))
    return [
        {"content": n["content"], "timestamp": n["timestamp"]}
        for n in user_notes
    ]

@router.post("/api/tasks")
async def save_tasks(tasks: list[TaskItem], current_user: dict = Depends(get_current_user)):
    if tasks_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    user_id = str(current_user["_id"])
    tasks_collection.delete_many({"user_id": user_id})
    for task in tasks:
        tasks_collection.insert_one({
            "user_id": user_id,
            "task": task.task,
            "checked": task.checked
        })
    return {"message": "Tasks saved successfully"}

@router.get("/api/tasks")
async def get_tasks(current_user: dict = Depends(get_current_user)):
    if tasks_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    user_id = str(current_user["_id"])
    user_tasks = list(tasks_collection.find({"user_id": user_id}))
    return [
        {"task": t["task"], "checked": t["checked"]}
        for t in user_tasks
    ]
