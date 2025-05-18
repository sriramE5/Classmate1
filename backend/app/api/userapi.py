from fastapi import APIRouter,FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, validator
from passlib.context import CryptContext
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import certifi
from jose import jwt, JWTError
from bson import ObjectId
from dotenv import load_dotenv # Ensure this import is present
import os
from datetime import datetime, timedelta

# ---------------- Load Environment Variables ----------------
load_dotenv() # Call after import
MONGO_URI = os.getenv("MONGO_URI")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

# ---------------- FastAPI Setup ----------------
router = APIRouter()

# CORS Middleware (Allow all for development)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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
    client.server_info()
    db = client["classmate"]
    users_collection = db["users"]
    print("✅ Connected to MongoDB Atlas with TLS")
except ConnectionFailure as e:
    print(f"❌ MongoDB Connection Error: {e}")
    db = None
    users_collection = None

# ---------------- Models ----------------
class RegisterModel(BaseModel):
    name: str
    email: str
    password: str
    dob: str  # "YYYY-MM-DD"

    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

class LoginModel(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    name: str
    email: str
    dob: str

# ---------------- Utils ----------------
def create_jwt_token(user_id: str) -> str:
    payload = {"id": user_id, "exp": datetime.utcnow() + timedelta(days=1)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if users_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("id")
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------------- Routes ----------------
@router.post("/api/register", response_model=UserResponse)
async def register(user: RegisterModel):
    if users_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = pwd_context.hash(user.password)
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": hashed_pw,
        "dob": user.dob,
        "created_at": datetime.utcnow()
    }
    users_collection.insert_one(user_data)

    return UserResponse(name=user.name, email=user.email, dob=user.dob)

@router.post("/api/login")
async def login(user: LoginModel):
    if users_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    db_user = users_collection.find_one({"email": user.email})
    if db_user is None or not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_jwt_token(str(db_user["_id"]))

    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "name": db_user["name"],
            "email": db_user["email"]
        }
    }

@router.get("/api/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        name=current_user["name"],
        email=current_user["email"],
        dob=current_user["dob"]
    )

@router.get("/")
async def health_check():
    return {
        "status": "OK",
        "database": "Connected" if users_collection is not None else "Disconnected"
    }