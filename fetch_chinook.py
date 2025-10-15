import os
import sqlite3
import sys
from pathlib import Path

import requests

CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)

def main():
    db_path = Path("chinook.db")
    if db_path.exists():
        print("chinook.db already exists. Nothing to do.")
        return

    print("Downloading Chinook SQL...")
    r = requests.get(CHINOOK_SQL_URL, timeout=60)
    r.raise_for_status()
    sql_text = r.text

    print("Creating chinook.db and loading data (this may take a moment)...")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql_text)
        conn.commit()
    finally:
        conn.close()

    print("Done! Created chinook.db with Chinook schema and sample data.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Failed to create chinook.db:", e)
        sys.exit(1)
