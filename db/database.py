from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file

DATABASE_URL = os.getenv("DATABASE_URL")

# engine = actual connection to postgres
engine = create_engine(DATABASE_URL)

# SessionLocal = factory to create DB sessions
SessionLocal = sessionmaker(bind=engine)

# Base = parent class for all our DB table models
Base = declarative_base()

# Dependency — gives a DB session to each route, closes it after
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()