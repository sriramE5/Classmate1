from fastapi import APIRouter, FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional
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
MONGO_URI = os.getenv("MONGO_URI")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
DEFAULT_AVATAR = "https://randomuser.me/api/portraits/lego/1.jpg" # A generic default

# ---------------- FastAPI Setup ----------------
router = APIRouter()

# ---------------- Security & Hashing Setup ----------------
# The tokenUrl is the path where the client will send username/password to get a token.
# If your main app includes this router with a prefix like /api, then this path is relative to that.
# However, given frontend calls like `${serverUrl}/api/me`, it implies /api/login is the full path.
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
    client.admin.command('ping') # Verify connection
    db = client["classmate"] # Use your actual database name
    users_collection = db["users"]
    print("✅ Connected to MongoDB Atlas with TLS")
except ConnectionFailure as e:
    print(f"❌ MongoDB Connection Error: {e}")
    db = None
    users_collection = None
except Exception as e:
    print(f"❌ An unexpected error occurred during MongoDB connection: {e}")
    db = None
    users_collection = None


# ---------------- Models ----------------
class RegisterModel(BaseModel):
    name: str
    email: EmailStr
    password: str
    dob: str  # "YYYY-MM-DD"

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
    avatar: Optional[str] = Field(default=DEFAULT_AVATAR)
    bio: Optional[str] = ""
    chatbot_customization_enabled: Optional[bool] = True
    chatbot_nickname: Optional[str] = ""
    chatbot_tone: Optional[str] = "friendly_supportive"
    chatbot_custom_instructions: Optional[str] = ""
    chatbot_user_context: Optional[str] = ""

class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    bio: Optional[str] = None
    avatar: Optional[str] = None
    chatbot_customization_enabled: Optional[bool] = None
    chatbot_nickname: Optional[str] = None
    chatbot_tone: Optional[str] = None
    chatbot_custom_instructions: Optional[str] = None
    chatbot_user_context: Optional[str] = None

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

    @validator("new_password")
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError("New password must be at least 6 characters long")
        return v

# ---------------- Utils ----------------
def create_jwt_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=1) # Default 1 day
    payload = {"sub": user_id, "exp": expire, "type": "access"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not connected")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise credentials_exception
    return user

# ---------------- Routes ----------------
# These routes assume they are the final paths, e.g. /api/register
@router.post("/api/register", response_model=UserResponse)
async def register(user: RegisterModel):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not connected")

    if users_collection.find_one({"email": user.email.lower()}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed_pw = pwd_context.hash(user.password)
    user_data = {
        "name": user.name,
        "email": user.email.lower(),
        "password": hashed_pw,
        "dob": user.dob,
        "avatar": DEFAULT_AVATAR,
        "bio": "",
        "chatbot_customization_enabled": True,
        "chatbot_nickname": "",
        "chatbot_tone": "friendly_supportive",
        "chatbot_custom_instructions": "",
        "chatbot_user_context": "",
        "created_at": datetime.utcnow()
    }
    result = users_collection.insert_one(user_data)
    created_user = users_collection.find_one({"_id": result.inserted_id})

    return UserResponse(
        name=created_user["name"],
        email=created_user["email"],
        dob=created_user["dob"],
        avatar=created_user.get("avatar", DEFAULT_AVATAR),
        bio=created_user.get("bio", ""),
        chatbot_customization_enabled=created_user.get("chatbot_customization_enabled", True),
        chatbot_nickname=created_user.get("chatbot_nickname", ""),
        chatbot_tone=created_user.get("chatbot_tone", "friendly_supportive"),
        chatbot_custom_instructions=created_user.get("chatbot_custom_instructions", ""),
        chatbot_user_context=created_user.get("chatbot_user_context", "")
    )

@router.post("/api/login")
async def login(user: LoginModel):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not connected")

    db_user = users_collection.find_one({"email": user.email.lower()})
    if db_user is None or not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_jwt_token(str(db_user["_id"]))

    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "name": db_user["name"],
            "email": db_user["email"],
            "avatar": db_user.get("avatar", DEFAULT_AVATAR)
        }
    }

@router.get("/api/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        name=current_user["name"],
        email=current_user["email"],
        dob=current_user["dob"],
        avatar=current_user.get("avatar", DEFAULT_AVATAR),
        bio=current_user.get("bio", ""),
        chatbot_customization_enabled=current_user.get("chatbot_customization_enabled", True),
        chatbot_nickname=current_user.get("chatbot_nickname", ""),
        chatbot_tone=current_user.get("chatbot_tone", "friendly_supportive"),
        chatbot_custom_instructions=current_user.get("chatbot_custom_instructions", ""),
        chatbot_user_context=current_user.get("chatbot_user_context", "")
    )

@router.patch("/api/me", response_model=UserResponse)
async def update_me(update_data: UserUpdateRequest, current_user: dict = Depends(get_current_user)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not connected")

    update_fields = update_data.model_dump(exclude_unset=True)

    if not update_fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided")

    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": update_fields}
    )

    updated_user_doc = users_collection.find_one({"_id": current_user["_id"]})
    
    return UserResponse(
        name=updated_user_doc["name"],
        email=updated_user_doc["email"],
        dob=updated_user_doc["dob"],
        avatar=updated_user_doc.get("avatar", DEFAULT_AVATAR),
        bio=updated_user_doc.get("bio", ""),
        chatbot_customization_enabled=updated_user_doc.get("chatbot_customization_enabled", True),
        chatbot_nickname=updated_user_doc.get("chatbot_nickname", ""),
        chatbot_tone=updated_user_doc.get("chatbot_tone", "friendly_supportive"),
        chatbot_custom_instructions=updated_user_doc.get("chatbot_custom_instructions", ""),
        chatbot_user_context=updated_user_doc.get("chatbot_user_context", "")
    )

@router.post("/api/password/change")
async def change_password_route(request: PasswordChangeRequest, current_user: dict = Depends(get_current_user)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not connected")

    if not pwd_context.verify(request.current_password, current_user["password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect current password")

    hashed_new_password = pwd_context.hash(request.new_password)
    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"password": hashed_new_password}}
    )
    return {"message": "Password changed successfully"}

@router.delete("/api/me")
async def delete_me_route(current_user: dict = Depends(get_current_user)):
    if users_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not connected")

    result = users_collection.delete_one({"_id": current_user["_id"]})
    if result.deleted_count == 1:
        return {"message": "Account deleted successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found for deletion")

# To run this file standalone for testing (if it's your main app file):
# if __name__ == "__main__":
#     app = FastAPI()
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"], # Allow all for development
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )
#     app.include_router(router) # No prefix, as routes in router already have /api
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
