"""Canonical data-directory paths used throughout Wintermute."""

from pathlib import Path

DATA_DIR = Path("data")

MEMORIES_FILE = DATA_DIR / "MEMORIES.txt"
SKILLS_DIR = DATA_DIR / "skills"
FTS5_DB_PATH = DATA_DIR / "memory_index.db"
SCHEDULER_DB = str(DATA_DIR / "scheduler.db")
