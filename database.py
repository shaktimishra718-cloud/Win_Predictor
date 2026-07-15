import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load variables from the local .env file (if running locally)
load_dotenv()

# Securely grab the single Neon connection string
DB_URL = os.getenv('DATABASE_URL')

def get_engine():
    """Creates and returns the SQLAlchemy engine."""
    if not DB_URL:
        print("❌ DATABASE_URL environment variable is missing!")
        return None
        
    try:
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return None

def test_connection():
    """Tests the connection to ensure credentials are correct."""
    engine = get_engine()
    if engine:
        try:
            with engine.connect() as connection:
                print("✅ Connection established successfully to the Cloud Database!")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    else:
        print("❌ Could not create engine.")

if __name__ == "__main__":
    test_connection()