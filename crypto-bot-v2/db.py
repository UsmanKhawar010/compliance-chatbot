
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('prompts_responses.db')
    c = conn.cursor()
    # Create table with created_at, prompt, and response columns
    c.execute('''
        CREATE TABLE IF NOT EXISTS prompts_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            prompt TEXT,
            response TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_prompt_response(prompt, response):
    conn = sqlite3.connect('prompts_responses.db')
    c = conn.cursor()
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
    
    # Insert prompt and response into the database
    c.execute("INSERT INTO prompts_responses (created_at, prompt, response) VALUES (?, ?, ?)",
              (created_at, prompt, response))
    
    conn.commit()
    conn.close()


