# app/api/tasksapi.py
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from bson import ObjectId
from datetime import datetime

from .userapi import get_current_user, db

router = APIRouter()
tasks_collection = db["tasks"]

class TaskInDB(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    text: str
    pinned: bool
    highlighted: bool
    checkbox_states: Dict[str, bool] = {}
    created_at: datetime

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}
        from_attributes = True

class TaskCreate(BaseModel):
    text: str
    pinned: bool = False
    highlighted: bool = False
    checkbox_states: Dict[str, bool] = {}

class TaskUpdate(BaseModel):
    text: Optional[str] = None
    pinned: Optional[bool] = None
    highlighted: Optional[bool] = None

class CheckboxStateUpdate(BaseModel):
    checkbox_states: Dict[str, bool]

@router.post("/api/tasks", response_model=List[TaskInDB], status_code=status.HTTP_201_CREATED)
async def create_tasks(tasks: List[TaskCreate], current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    new_tasks_data = []
    for task in tasks:
        task_data = task.model_dump()
        task_data["user_id"] = user_id
        task_data["created_at"] = datetime.utcnow()
        new_tasks_data.append(task_data)

    if not new_tasks_data:
        return []
        
    result = tasks_collection.insert_many(new_tasks_data)
    created_tasks = tasks_collection.find({"_id": {"$in": result.inserted_ids}})
    return [TaskInDB.model_validate({**task, "_id": str(task["_id"])}) for task in created_tasks]

@router.get("/api/tasks", response_model=List[TaskInDB])
async def get_tasks(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    user_tasks = tasks_collection.find({"user_id": user_id})
    return [TaskInDB.model_validate({**task, "_id": str(task["_id"])}) for task in user_tasks]

@router.put("/api/tasks/{task_id}", response_model=TaskInDB)
async def update_task(task_id: str, task_update: TaskUpdate, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID")

    update_data = task_update.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")

    result = tasks_collection.update_one(
        {"_id": ObjectId(task_id), "user_id": user_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    
    updated_task = tasks_collection.find_one({"_id": ObjectId(task_id)})
    return TaskInDB.model_validate({**updated_task, "_id": str(updated_task["_id"])})

@router.patch("/api/tasks/{task_id}/states", response_model=TaskInDB)
async def update_task_states(task_id: str, state_update: CheckboxStateUpdate, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID")

    result = tasks_collection.update_one(
        {"_id": ObjectId(task_id), "user_id": user_id},
        {"$set": {"checkbox_states": state_update.checkbox_states}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
        
    updated_task = tasks_collection.find_one({"_id": ObjectId(task_id)})
    return TaskInDB.model_validate({**updated_task, "_id": str(updated_task["_id"])})


@router.delete("/api/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: str, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    if not ObjectId.is_valid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID")

    delete_result = tasks_collection.delete_one({"_id": ObjectId(task_id), "user_id": user_id})
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return
