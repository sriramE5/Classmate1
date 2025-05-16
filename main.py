from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
import markdown2

# Load environment variables
load_dotenv()

# Setup Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Initialize FastAPI
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    prompt: str

# Chat endpoint
@app.post("/api/chat")
async def chat(req: ChatRequest, as_markdown: bool = Query(False)):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=req.prompt
        )
        reply_text = response.text

        if as_markdown:
            reply_text = markdown2.markdown(reply_text)

        return {"reply": reply_text}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


