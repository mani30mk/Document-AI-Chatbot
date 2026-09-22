-- Migration: Create chat_sessions table for persistent session state
-- Run this in your Supabase SQL Editor: https://supabase.com/dashboard/project/_/sql

create table if not exists chat_sessions (
  session_id uuid primary key,
  files jsonb not null default '[]',
  docs jsonb not null default '{}',
  slides jsonb not null default '{}',
  history jsonb not null default '[]',
  updated_at timestamptz not null default now()
);

-- Index on updated_at for cleanup or sorting if needed
create index if not exists idx_chat_sessions_updated_at on chat_sessions (updated_at desc);
