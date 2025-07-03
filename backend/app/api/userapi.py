# app/api/userapi.py

from fastapi import APIRouter,FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, validator, EmailStr
from passlib.context import CryptContext
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import certifi
from jose import jwt, JWTError
from bson import ObjectId
from dotenv import load_dotenv 
import os
from datetime import datetime, timedelta

# ---------------- Load Environment Variables ----------------
load_dotenv() 
# MONGO_URI = os.getenv("MONGO_URI")
# SECRET_KEY = os.getenv("SECRET_KEY")
MONGO_URI = os.environ.get('MONGO_URI')
SECRET_KEY = os.environ.get('SECRET_KEY')
ALGORITHM = "HS256"

# ---------------- FastAPI Setup ----------------
router = APIRouter()

# ---------------- Security & Hashing Setup ----------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login") 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- MongoDB Connection ----------------
try:
    print(f"📡 Connecting to MongoDB at URI: {MONGO_URI}")
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tlsCAFile=certifi.where()
    )
    client.admin.command('ping') 
    db = client["classmate"] 
    users_collection = db["users"] 
    print("✅ Connected to MongoDB Atlas with TLS")
except ConnectionFailure as e:
    print(f"❌ MongoDB Connection Error: {e}")
    db = None
    users_collection = None
except Exception as e:
    print(f"❌ An unexpected error occurred during MongoDB setup: {e}")
    db = None
    users_collection = None


# ---------------- Models ----------------
class RegisterModel(BaseModel):
    name: str
    email: EmailStr 
    password: str
    dob: str

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 6: 
            raise ValueError("Password must be at least 6 characters")
        return v

class LoginModel(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    name: str
    email: EmailStr
    dob: str
    bio: str | None = None
    avatar: str | None = None
    chatbot_customization_enabled: bool | None = True # New field
    chatbot_nickname: str | None = ""
    chatbot_tone: str | None = "friendly_supportive"
    chatbot_custom_instructions: str | None = ""
    chatbot_user_context: str | None = ""


class UpdateProfileModel(BaseModel):
    name: str | None = None
    bio: str | None = None
    avatar: str | None = None
    chatbot_customization_enabled: bool | None = None # New field
    chatbot_nickname: str | None = None
    chatbot_tone: str | None = None
    chatbot_custom_instructions: str | None = None
    chatbot_user_context: str | None = None


class ChangePasswordModel(BaseModel):
    current_password: str
    new_password: str

    @validator("new_password")
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError("New password must be at least 6 characters")
        return v

# ---------------- Utils ----------------
def create_jwt_token(user_id: str) -> str:
    payload = {"id": user_id, "exp": datetime.utcnow() + timedelta(days=1)} 
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service not available")
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str | None = payload.get("id")
        if user_id_str is None:
            raise credentials_exception
        user_id = ObjectId(user_id_str) 
    except JWTError:
        raise credentials_exception
    except Exception: 
        raise credentials_exception
        
    user = users_collection.find_one({"_id": user_id})
    if user is None:
        raise credentials_exception
    return user

# ---------------- Routes ----------------
@router.post("/api/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: RegisterModel):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service not available")

    if users_collection.find_one({"email": user.email.lower()}): 
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed_pw = pwd_context.hash(user.password)
    user_data = {
        "name": user.name,
        "email": user.email.lower(),
        "password": hashed_pw,
        "dob": user.dob,
        "bio": "", 
        "avatar": "", 
        "chatbot_customization_enabled": True, # Default to True
        "chatbot_nickname": "",
        "chatbot_tone": "friendly_supportive",
        "chatbot_custom_instructions": "",
        "chatbot_user_context": "",
        "created_at": datetime.utcnow()
    }
    result = users_collection.insert_one(user_data)
    
    created_user = users_collection.find_one({"_id": result.inserted_id})
    if not created_user: 
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve created user")

    return UserResponse(
        name=created_user["name"], 
        email=created_user["email"], 
        dob=created_user["dob"],
        bio=created_user.get("bio", ""),
        avatar=created_user.get("avatar", ""),
        chatbot_customization_enabled=created_user.get("chatbot_customization_enabled", True),
        chatbot_nickname=created_user.get("chatbot_nickname", ""),
        chatbot_tone=created_user.get("chatbot_tone", "friendly_supportive"),
        chatbot_custom_instructions=created_user.get("chatbot_custom_instructions", ""),
        chatbot_user_context=created_user.get("chatbot_user_context", "")
    )

@router.post("/api/login")
async def login(form_data: LoginModel): 
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service not available")

    db_user = users_collection.find_one({"email": form_data.email.lower()})
    if db_user is None or not pwd_context.verify(form_data.password, db_user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_jwt_token(str(db_user["_id"]))

    return {
        "message": "Login successful",
        "token": token,
        "token_type": "bearer", 
        "user": {
            "name": db_user["name"],
            "email": db_user["email"],
            "bio": db_user.get("bio", ""),
            "avatar": db_user.get("avatar", ""),
            "chatbot_customization_enabled": db_user.get("chatbot_customization_enabled", True),
            "chatbot_nickname": db_user.get("chatbot_nickname", ""),
            "chatbot_tone": db_user.get("chatbot_tone", "friendly_supportive"),
            "chatbot_custom_instructions": db_user.get("chatbot_custom_instructions", ""),
            "chatbot_user_context": db_user.get("chatbot_user_context", "")
        }
    }

@router.get("/api/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        name=current_user.get("name"),
        email=current_user.get("email"),
        dob=current_user.get("dob"),
        bio=current_user.get("bio", ""),
        avatar=current_user.get("avatar", ""),
        chatbot_customization_enabled=current_user.get("chatbot_customization_enabled", True),
        chatbot_nickname=current_user.get("chatbot_nickname", ""),
        chatbot_tone=current_user.get("chatbot_tone", "friendly_supportive"),
        chatbot_custom_instructions=current_user.get("chatbot_custom_instructions", ""),
        chatbot_user_context=current_user.get("chatbot_user_context", "")
    )

@router.patch("/api/me", response_model=UserResponse)
async def update_me(profile_data: UpdateProfileModel, current_user: dict = Depends(get_current_user)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service not available")

    update_fields = profile_data.model_dump(exclude_unset=True) 

    if not update_fields:
         return UserResponse(
            name=current_user.get("name"), email=current_user.get("email"), dob=current_user.get("dob"),
            bio=current_user.get("bio", ""), avatar=current_user.get("avatar", ""),
            chatbot_customization_enabled=current_user.get("chatbot_customization_enabled", True),
            chatbot_nickname=current_user.get("chatbot_nickname", ""),
            chatbot_tone=current_user.get("chatbot_tone", "friendly_supportive"),
            chatbot_custom_instructions=current_user.get("chatbot_custom_instructions", ""),
            chatbot_user_context=current_user.get("chatbot_user_context", "")
        )

    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": update_fields}
    )

    updated_user_doc = users_collection.find_one({"_id": current_user["_id"]})
    return UserResponse(
        name=updated_user_doc.get("name"),
        email=updated_user_doc.get("email"),
        dob=updated_user_doc.get("dob"),
        bio=updated_user_doc.get("bio", ""),
        avatar=updated_user_doc.get("avatar", ""),
        chatbot_customization_enabled=updated_user_doc.get("chatbot_customization_enabled", True),
        chatbot_nickname=updated_user_doc.get("chatbot_nickname", ""),
        chatbot_tone=updated_user_doc.get("chatbot_tone", "friendly_supportive"),
        chatbot_custom_instructions=updated_user_doc.get("chatbot_custom_instructions", ""),
        chatbot_user_context=updated_user_doc.get("chatbot_user_context", "")
    )

# ... (rest of userapi.py remains the same: change_password, delete_me, health_check) ...

@router.post("/api/password/change")
async def change_password(password_data: ChangePasswordModel, current_user: dict = Depends(get_current_user)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service not available")

    if not pwd_context.verify(password_data.current_password, current_user["password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect current password")

    hashed_new_password = pwd_context.hash(password_data.new_password)
    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"password": hashed_new_password}}
    )
    return {"message": "Password updated successfully"}

@router.delete("/api/me", status_code=status.HTTP_200_OK)
async def delete_me(current_user: dict = Depends(get_current_user)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service not available")

    delete_result = users_collection.delete_one({"_id": current_user["_id"]})
    
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found for deletion")
        
    return {"message": "Account deleted successfully"}

@router.get("/api/health") 
async def health_check():
    db_status = "Disconnected"
    if db and users_collection is not None:
        try:
            client.admin.command('ping') 
            db_status = "Connected"
        except Exception:
            db_status = "Connection Error"
            
    return {
        "status": "OK",
        "database": db_status
    }
