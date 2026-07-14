from database import get_engine
from sqlalchemy import text

def wipe_table():
    print("Connecting to PostgreSQL...")
    engine = get_engine()
    
    with engine.connect() as conn:
        # TRUNCATE deletes all rows but keeps the columns/structure intact
        conn.execute(text("TRUNCATE TABLE matches;"))
        conn.commit()
        
    print("Success! The matches table is now completely empty.")

if __name__ == "__main__":
    wipe_table()