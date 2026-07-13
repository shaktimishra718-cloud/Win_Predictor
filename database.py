import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Build the connection string
# Format: postgresql://username:password@host:port/database
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME')

DB_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

def get_engine():
    """Creates and returns the SQLAlchemy engine."""
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
                print("✅ Connection established successfully!")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    else:
        print("❌ Could not create engine.")

if __name__ == "__main__":
    test_connection()