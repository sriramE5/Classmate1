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

# Use a simple try-except block to handle potential initialization errors gracefully
try:
    client1 = genai.Client(api_key=api_key1)
except Exception as e:
    print(f"Warning: Could not initialize Gemini client1. Error: {e}")
    client1 = None

try:
    client2 = genai.Client(api_key=api_key2)
except Exception as e:
    print(f"Warning: Could not initialize Gemini client2. Error: {e}")
    client2 = None


router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str

async def generate_reply(prompt: str, client):
    if not client:
        raise Exception("Gemini client not initialized.")
    response = client.models.generate_content(
        model="gemini-2.0-flash", # Assuming this is the correct model name
        contents=prompt
    )
    return response.text

@router.post("/api/chat")
async def chat(req: ChatRequest, as_markdown: bool = Query(False)):
    try:
        reply_text = ""
        # First try with client1
        try:
            if client1:
                reply_text = await generate_reply(req.prompt, client1)
            else:
                raise Exception("Client1 not available")
        except Exception as e1:
            # If error (e.g., 503, client unavailable), switch to client2
            print(f"Client1 failed with error: {e1}. Switching to Client2.")
            try:
                if client2:
                    reply_text = await generate_reply(req.prompt, client2)
                else:
                     raise Exception("Client2 not available")
            except Exception as e2:
                print(f"Client2 also failed with error: {e2}.")
                return {"reply": "Service is temporarily busy. Please try again later."}

        if as_markdown:
            reply_text = markdown2.markdown(reply_text)
        return {"reply": reply_text}
    except Exception as e:
        return {"reply": f"An unexpected error occurred: {str(e)}"}
