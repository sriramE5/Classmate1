from fastapi import APIRouter, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
import markdown2

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
