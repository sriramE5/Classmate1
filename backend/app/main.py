from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import userapi, chatbotapi 

app = FastAPI()

# Enable CORS (adjust origins in production)
# This is the CORRECT place to define your global CORS policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, "*" is okay. For production, specify exact origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(userapi.router)
app.include_router(chatbotapi.router) # This router should NOT have its own CORS middleware anymore

@app.get("/")
async def root():
    return {"message": "Server is running"}
