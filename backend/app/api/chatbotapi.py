from fastapi import APIRouter, FastAPI, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
import markdown2
import json
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.")
    # Handle the case where API key is missing, maybe raise an error or use a default
    # For now, let's proceed but Gemini calls will fail.
    client = None 
else:
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model_name="gemini-1.5-flash") # Use a valid model

# Initialize FastAPI router
router = APIRouter()

# --- Data Structures (Pydantic Models) ---
class ChatMessage(BaseModel):
    role: str # "user" or "bot"
    content: str
    timestamp: str | None = None # Add timestamp

class ChatSession(BaseModel):
    id: str
    title: str | None = "New Chat"
    messages: list[ChatMessage] = []
    created_at: str
    last_updated_at: str

class ChatRequest(BaseModel):
    prompt: str
    chat_id: str | None = None

class NewChatResponse(BaseModel):
    chat_id: str
    title: str
    created_at: str
    last_updated_at: str

class RenameRequest(BaseModel):
    chat_id: str
    new_title: str

# --- In-memory storage (replace with DB or file in production) ---
# Store chats as a dictionary where key is chat_id and value is ChatSession object
chats: dict[str, ChatSession] = {}

# --- Helper Functions ---
def get_current_timestamp():
    return datetime.now().isoformat()

# --- API Endpoints ---

@router.post("/new_chat", response_model=NewChatResponse)
async def create_new_chat():
    """Creates a new chat session and returns its ID and initial details."""
    new_chat_id = str(uuid.uuid4())
    timestamp = get_current_timestamp()
    new_chat_session = ChatSession(
        id=new_chat_id,
        title="New Chat", # Default title
        messages=[],
        created_at=timestamp,
        last_updated_at=timestamp
    )
    chats[new_chat_id] = new_chat_session
    print(f"Created new chat: {new_chat_id}")
    return NewChatResponse(
        chat_id=new_chat_id,
        title=new_chat_session.title,
        created_at=new_chat_session.created_at,
        last_updated_at=new_chat_session.last_updated_at
    )

@router.post("/chat")
async def chat(req: ChatRequest):
    """Handles incoming chat messages, generates responses, and saves conversation."""
    chat_id = req.chat_id
    user_prompt = req.prompt
    timestamp = get_current_timestamp()

    print(f"Received request for chat_id: {chat_id}")

    # --- Get or Create Chat Session ---
    if not chat_id or chat_id not in chats:
        # This case should ideally not happen if frontend always calls /new_chat first
        # But handle it defensively: create a new chat if ID is missing or invalid
        print(f"Warning: chat_id '{chat_id}' not found or missing. Creating a new chat.")
        new_chat_response = await create_new_chat() # Create a new chat
        chat_id = new_chat_response.chat_id
        # Add a placeholder message indicating creation?
        # Or just proceed with the user's prompt in the new chat

    chat_session = chats[chat_id]

    # --- Add user message to history ---
    user_message = ChatMessage(role="user", content=user_prompt, timestamp=timestamp)
    chat_session.messages.append(user_message)
    chat_session.last_updated_at = timestamp
    print(f"Added user message to chat {chat_id}")

    # --- Generate bot response using Gemini --- 
    bot_reply_text = "Error: Could not get response."
    if client:
        try:
            # Construct history for Gemini (optional, depends on model needs)
            # history_for_gemini = [{"role": msg.role, "parts": [msg.content]} for msg in chat_session.messages[:-1]] # Exclude current prompt
            
            # Simple generation for now
            response = client.generate_content(user_prompt)
            # Check for safety ratings if necessary
            # if response.prompt_feedback.block_reason:
            #     bot_reply_text = f"Blocked due to: {response.prompt_feedback.block_reason}"
            # else:
            bot_reply_text = response.text
            print(f"Generated bot response for chat {chat_id}")

        except Exception as e:
            print(f"Error generating response: {e}")
            bot_reply_text = f"Error: {str(e)}"
    else:
        bot_reply_text = "Error: Gemini client not initialized (API key missing?)."

    # --- Add bot response to history ---
    bot_message = ChatMessage(role="bot", content=bot_reply_text, timestamp=get_current_timestamp())
    chat_session.messages.append(bot_message)
    chat_session.last_updated_at = get_current_timestamp() # Update again after bot response
    print(f"Added bot message to chat {chat_id}")

    # --- Update chat title based on first user prompt if it's still "New Chat" ---
    if chat_session.title == "New Chat" and len(chat_session.messages) >= 2: # After user and bot message
        try:
            # Generate a title using a separate prompt (or use first few words)
            # title_prompt = f"Generate a short title (3-5 words) for this conversation start: User: {user_prompt}\nBot: {bot_reply_text}"
            # title_response = client.generate_content(title_prompt)
            # chat_session.title = title_response.text.strip()
            # Simple title for now:
            chat_session.title = user_prompt[:30] + ("..." if len(user_prompt) > 30 else "")
            print(f"Updated title for chat {chat_id} to: {chat_session.title}")
        except Exception as e:
            print(f"Error generating title: {e}")
            # Keep default title if generation fails

    # --- Return bot response --- 
    # The frontend expects { "reply": "..." }
    # We should also return the updated title if it changed
    return {
        "reply": bot_reply_text,
        "chat_id": chat_id, # Return chat_id in case a new one was created
        "title": chat_session.title # Return potentially updated title
    }

@router.get("/history")
async def get_history():
    """Returns all chat sessions (metadata and messages)."""
    # Convert ChatSession objects to dictionaries for JSON serialization
    history_data = {chat_id: session.dict() for chat_id, session in chats.items()}
    print(f"Returning history for {len(history_data)} chats.")
    return history_data

@router.post("/rename_chat")
async def rename_chat(req: RenameRequest):
    """Renames a specific chat session."""
    chat_id = req.chat_id
    new_title = req.new_title
    if chat_id in chats:
        chats[chat_id].title = new_title
        chats[chat_id].last_updated_at = get_current_timestamp()
        print(f"Renamed chat {chat_id} to: {new_title}")
        return {"message": "Chat renamed successfully", "chat_id": chat_id, "new_title": new_title}
    else:
        print(f"Error: Chat ID {chat_id} not found for renaming.")
        raise HTTPException(status_code=404, detail="Chat not found")

@router.delete("/delete_chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Deletes a specific chat session."""
    if chat_id in chats:
        del chats[chat_id]
        print(f"Deleted chat: {chat_id}")
        return {"message": "Chat deleted successfully", "chat_id": chat_id}
    else:
        print(f"Error: Chat ID {chat_id} not found for deletion.")
        raise HTTPException(status_code=404, detail="Chat not found")

# Note: Removed /api/goals and /api/performance endpoints as they seem unrelated
# to the core chatbot functionality described in the frontend files.
# If they are needed, they should be reviewed and potentially updated separately.


