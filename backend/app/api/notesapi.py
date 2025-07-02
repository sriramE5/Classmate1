# app/api/notesapi.py
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional
from bson import ObjectId
from datetime import datetime

# Import shared dependencies from userapi
from .userapi import get_current_user, db

router = APIRouter()
notes_collection = db["notes"]

class ListItem(BaseModel):
    text: str
    checked: bool

class Attachment(BaseModel):
    name: str
    type: str
    data: str

class NoteInDB(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    title: str
    type: str
    content: Optional[str] = ""
    listItems: Optional[List[ListItem]] = []
    attachments: Optional[List[Attachment]] = []
    pinned: bool = False
    highlighted: bool = False
    timestamp: datetime

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}
        from_attributes = True

class NoteCreateUpdate(BaseModel):
    title: str
    type: str
    content: Optional[str] = ""
    listItems: Optional[List[ListItem]] = []
    attachments: Optional[List[Attachment]] = []
    pinned: bool = False
    highlighted: bool = False
    # REMOVED: timestamp: datetime - The frontend will no longer send this.

@router.post("/api/notes", response_model=NoteInDB, status_code=status.HTTP_201_CREATED)
async def create_note(note: NoteCreateUpdate, current_user: dict = Depends(get_current_user)):
    note_data = note.model_dump()
    note_data["user_id"] = str(current_user["_id"])
    note_data["timestamp"] = datetime.utcnow()  # ADDED: Server generates the timestamp
    
    result = notes_collection.insert_one(note_data)
    created_note = notes_collection.find_one({"_id": result.inserted_id})

    # ADDED: Important check to prevent errors if the note isn't found
    if not created_note:
        raise HTTPException(status_code=500, detail="Failed to create and retrieve note.")
        
    created_note["_id"] = str(created_note["_id"])
    return NoteInDB.model_validate(created_note)


@router.get("/api/notes", response_model=List[NoteInDB])
async def get_notes(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    user_notes = notes_collection.find({"user_id": user_id})
    return [
        NoteInDB.model_validate({**note, "_id": str(note["_id"])})    
        for note in user_notes 
    ]

@router.put("/api/notes/{note_id}", response_model=NoteInDB)
async def update_note(note_id: str, note_update: NoteCreateUpdate, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid note ID")
    
    existing_note = notes_collection.find_one({"_id": ObjectId(note_id), "user_id": user_id})
    if not existing_note:
        raise HTTPException(status_code=404, detail="Note not found")

    update_data = note_update.model_dump()
    update_data["timestamp"] = datetime.utcnow() # ADDED: Update timestamp on every edit

    notes_collection.update_one({"_id": ObjectId(note_id)}, {"$set": update_data})
    updated_note = notes_collection.find_one({"_id": ObjectId(note_id)})

    if not updated_note:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated note.")

    updated_note["_id"] = str(updated_note["_id"])
    return NoteInDB.model_validate(updated_note)


@router.delete("/api/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(note_id: str, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid note ID")

    delete_result = notes_collection.delete_one({"_id": ObjectId(note_id), "user_id": user_id})
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    return
