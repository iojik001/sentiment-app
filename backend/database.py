import os
import sqlite3
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "data", "history.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                time_ms INTEGER,
                created_at TEXT NOT NULL
            )
        """)


def save_analysis(text, label, confidence, time_ms):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO analyses (text, label, confidence, time_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (text, label, confidence, time_ms, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def get_history(limit=20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, text, label, confidence, time_ms, created_at "
            "FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_history():
    with get_conn() as conn:
        conn.execute("DELETE FROM analyses")