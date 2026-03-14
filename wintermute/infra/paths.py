"""Canonical data-directory paths used throughout Wintermute."""

from pathlib import Path

DATA_DIR = Path("data")

SKILLS_DIR = DATA_DIR / "skills"
PROMPTS_DIR = DATA_DIR / "prompts"
# Legacy FTS5 paths — kept for reference; backends removed.
# FTS5_DB_PATH = DATA_DIR / "memory_index.db"
# SKILLS_FTS5_DB_PATH = DATA_DIR / "skill_index.db"
SKILLS_VECTOR_DB_PATH = DATA_DIR / "skill_vectors.db"
SCHEDULER_DB = DATA_DIR / "scheduler.db"
HOOKS_FILE = DATA_DIR / "CONVERGENCE_PROTOCOL_HOOKS.txt"
HOOKS_FILE_LEGACY = DATA_DIR / "TURING_PROTOCOL_HOOKS.txt"
GEMINI_CREDENTIALS_FILE = DATA_DIR / "gemini_credentials.json"
KIMI_CREDENTIALS_FILE = DATA_DIR / "kimi_credentials.json"
KIMI_DEVICE_ID_FILE = DATA_DIR / ".kimi_device_id"
