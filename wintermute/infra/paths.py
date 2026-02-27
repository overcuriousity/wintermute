"""Canonical data-directory paths used throughout Wintermute."""

from pathlib import Path

DATA_DIR = Path("data")

MEMORIES_FILE = DATA_DIR / "MEMORIES.txt"
SKILLS_DIR = DATA_DIR / "skills"
PROMPTS_DIR = DATA_DIR / "prompts"
FTS5_DB_PATH = DATA_DIR / "memory_index.db"
SCHEDULER_DB = str(DATA_DIR / "scheduler.db")
HOOKS_FILE = DATA_DIR / "TURING_PROTOCOL_HOOKS.txt"
GEMINI_CREDENTIALS_FILE = DATA_DIR / "gemini_credentials.json"
KIMI_CREDENTIALS_FILE = DATA_DIR / "kimi_credentials.json"
KIMI_DEVICE_ID_FILE = DATA_DIR / ".kimi_device_id"
