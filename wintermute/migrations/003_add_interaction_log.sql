CREATE TABLE IF NOT EXISTS interaction_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    action      TEXT    NOT NULL,
    session     TEXT    NOT NULL,
    llm         TEXT    NOT NULL,
    input       TEXT    NOT NULL,
    output      TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'ok'
);
