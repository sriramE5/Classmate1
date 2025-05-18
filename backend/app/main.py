from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import userapi, chatbotapi

app = FastAPI()

# Enable CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(userapi.router, prefix="/api")
app.include_router(chatbotapi.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Server is running"}
