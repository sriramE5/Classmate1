import os
from dotenv import load_dotenv 
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
import motor.motor_asyncio
from fastapi.encoders import jsonable_encoder

from .userapi import get_current_user

router = APIRouter()
load_dotenv()

# MongoDB connection setup
MONGO_DETAILS = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = client["classmate"]
chats_collection = db["chats"]

class Message(BaseModel):
    sender: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatCreate(BaseModel):
    title: str = "New Chat"
    user_id: Optional[str] = None  

class ChatUpdate(BaseModel):
    title: Optional[str] = None

class ChatInDB(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    title: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}

class ChatInfo(BaseModel):
    id: str = Field(alias="_id")
    title: str
    updated_at: datetime
    preview: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}

async def get_chat(chat_id: str, user_id: str):
    if not ObjectId.is_valid(chat_id):
        return None
    chat = await chats_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    if chat:
        chat["_id"] = str(chat["_id"])
    return chat

@router.post("/api/chats", response_model=ChatInDB, status_code=status.HTTP_201_CREATED)
async def create_chat(chat_data: ChatCreate, current_user: dict = Depends(get_current_user)):
    chat_data = {
        "user_id": str(current_user["_id"]),
        "title": chat_data.title,
        "messages": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = await chats_collection.insert_one(chat_data)
    new_chat = await chats_collection.find_one({"_id": result.inserted_id})

    if not new_chat:
        raise HTTPException(status_code=500, detail="Failed to create chat")
    
    new_chat["_id"] = str(new_chat["_id"])
    return ChatInDB(**new_chat)

@router.get("/api/chats", response_model=List[ChatInfo])
async def get_all_chats(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    chats = []
    async for chat in chats_collection.find({"user_id": user_id}).sort("updated_at", -1):
        chat["_id"] = str(chat["_id"])
        preview = None
        if chat.get("messages"):
            last_msg = chat["messages"][-1]
            preview = last_msg["content"][:50] + "..." if len(last_msg["content"]) > 50 else last_msg["content"]
        chats.append({
            "_id": chat["_id"],
            "title": chat["title"],
            "updated_at": chat.get("updated_at", chat.get("created_at", datetime.utcnow())),
            "preview": preview
        })
    return chats

@router.get("/api/chats/{chat_id}", response_model=ChatInDB)
async def get_chat_by_id(chat_id: str, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    chat = await get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatInDB(**chat)

@router.post("/api/chats/{chat_id}/messages", response_model=ChatInDB)
async def add_message(
    chat_id: str, 
    message: Message, 
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")
    
    message_data = message.dict()
    message_data["timestamp"] = datetime.utcnow()
    
    update_result = await chats_collection.update_one(
        {"_id": ObjectId(chat_id), "user_id": user_id},
        {
            "$push": {"messages": message_data},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    
    if update_result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    updated_chat = await get_chat(chat_id, user_id)
    return ChatInDB(**updated_chat)

@router.put("/api/chats/{chat_id}", response_model=ChatInDB)
async def update_chat(
    chat_id: str, 
    chat_update: ChatUpdate, 
    current_user: dict = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")
    
    update_data = {"updated_at": datetime.utcnow()}
    if chat_update.title is not None:
        update_data["title"] = chat_update.title
    
    update_result = await chats_collection.update_one(
        {"_id": ObjectId(chat_id), "user_id": user_id},
        {"$set": update_data}
    )
    
    if update_result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    updated_chat = await get_chat(chat_id, user_id)
    return ChatInDB(**updated_chat)

@router.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")
    
    delete_result = await chats_collection.delete_one(
        {"_id": ObjectId(chat_id), "user_id": user_id}
    )
    
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")

@router.delete("/api/chats", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_chats(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    await chats_collection.delete_many({"user_id": user_id})
