/*
  # Add decision integrity validation

  The decisions table references winner_exp_id and loser_exp_id as plain
  text with no FK constraint, because experiment_runs uses a composite unique
  key (user_id, exp_id) — not a standalone UNIQUE(exp_id). This means you
  can record a decision for an experiment that doesn't exist.

  This migration adds a trigger that validates both exp_id references
  belong to the same user before inserting a decision row.

  Also adds a partial index on experiment_runs(exp_id, user_id) to make
  the trigger lookup fast.
*/

-- Fast lookup index for the trigger
CREATE INDEX IF NOT EXISTS idx_exp_runs_user_exp
  ON experiment_runs(user_id, exp_id);

-- Validation function: winner_exp_id must exist for this user.
-- loser_exp_id is optional — skip validation when NULL.
CREATE OR REPLACE FUNCTION validate_decision_exp_ids()
RETURNS trigger AS $$
BEGIN
  -- winner is required
  IF NOT EXISTS (
    SELECT 1 FROM experiment_runs
    WHERE user_id = NEW.user_id
      AND exp_id  = NEW.winner_exp_id
  ) THEN
    RAISE EXCEPTION
      'Decision references unknown winner_exp_id "%" for this user. '
      'Create the experiment first.',
      NEW.winner_exp_id;
  END IF;

  -- loser is optional
  IF NEW.loser_exp_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM experiment_runs
    WHERE user_id = NEW.user_id
      AND exp_id  = NEW.loser_exp_id
  ) THEN
    RAISE EXCEPTION
      'Decision references unknown loser_exp_id "%" for this user. '
      'Create the experiment first.',
      NEW.loser_exp_id;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Attach trigger to decisions table (INSERT only — decisions are immutable)
DROP TRIGGER IF EXISTS trg_validate_decision_exp_ids ON decisions;
CREATE TRIGGER trg_validate_decision_exp_ids
  BEFORE INSERT ON decisions
  FOR EACH ROW EXECUTE FUNCTION validate_decision_exp_ids();
