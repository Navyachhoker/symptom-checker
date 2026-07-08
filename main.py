from fastapi import FastAPI
from db.database import Base, engine
from routes.chat import router as chat_router
from routes.history import router as history_router

# Create all DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Mini Symptom Checker")

# Register routes
app.include_router(chat_router)
app.include_router(history_router)

@app.get("/")
def root():
    return {"message": "Symptom Checker API is running"}