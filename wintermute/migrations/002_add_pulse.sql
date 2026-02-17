CREATE TABLE IF NOT EXISTS pulse (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    content   TEXT    NOT NULL,
    status    TEXT    NOT NULL DEFAULT 'active',
    priority  INTEGER NOT NULL DEFAULT 5,
    created   REAL    NOT NULL,
    updated   REAL,
    thread_id TEXT
);
