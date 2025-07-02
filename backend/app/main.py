from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import userapi, chatbotapi, notesapi, tasksapi, chathistoryapi

app = FastAPI()

# Enable CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(userapi.router)
app.include_router(chatbotapi.router)
app.include_router(notesapi.router)
app.include_router(tasksapi.router)
app.include_router(chathistoryapi.router)


@app.get("/")
async def root():
    return {"message": "Server is running"}
