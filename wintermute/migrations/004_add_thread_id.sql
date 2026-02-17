ALTER TABLE messages ADD COLUMN thread_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE summaries ADD COLUMN thread_id TEXT NOT NULL DEFAULT 'default';
