# init_db.py
import sqlite3
from datetime import datetime

DB_PATH = "faqs.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# جدول الأسئلة الشائعة
c.execute("""
CREATE TABLE IF NOT EXISTS faqs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer   TEXT NOT NULL,
    category TEXT,
    created_at TEXT
)
""")

# جدول المحادثات
c.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp  TEXT,
    user_message TEXT,
    bot_reply    TEXT,
    matched_faq_id INTEGER,
    confidence REAL
)
""")

conn.commit()
conn.close()
print("✅ Database initialized (faqs.db)")
