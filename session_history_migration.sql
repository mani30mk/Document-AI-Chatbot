-- Migration: Add device_id to chat_sessions table for ChatGPT-style multi-session history
-- Run this in your Supabase SQL Editor

alter table chat_sessions add column if not exists device_id text;
create index if not exists idx_chat_sessions_device_id on chat_sessions (device_id);
create index if not exists idx_chat_sessions_updated_at on chat_sessions (updated_at desc);
