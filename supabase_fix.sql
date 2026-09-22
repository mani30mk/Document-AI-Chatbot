-- Fix: Ensure documents table has vector(384) column matching FastEmbed (bge-small-en-v1.5) output
-- Run this in your Supabase SQL Editor: https://supabase.com/dashboard/project/_/sql
--
-- IMPORTANT: If you previously used GoogleGenerativeAIEmbeddings (text-embedding-004, 768-dim),
-- your documents table may have vector(768). Since we now use FastEmbed (384-dim) via the
-- remote embedding service, the column must be vector(384).
--
-- This will DELETE all existing vector data (which is fine since dimension-mismatched data
-- is unusable anyway).

-- Step 1: Clear any dimension-mismatched data
truncate table documents;

-- Step 2: Ensure column is vector(384) for FastEmbed bge-small-en-v1.5
alter table documents alter column embedding type vector(384);
