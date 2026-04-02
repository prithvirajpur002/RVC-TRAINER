-- Run this once in your Supabase SQL editor.
-- Adds the rvc_commit column to track which RVC version each experiment used.
ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS rvc_commit TEXT;
