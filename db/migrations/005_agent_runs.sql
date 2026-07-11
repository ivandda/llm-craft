CREATE TABLE IF NOT EXISTS agent_runs (
  id text PRIMARY KEY,
  model text NOT NULL,
  goal_id text NOT NULL,
  goal_title text NOT NULL,
  target text NOT NULL,
  requested_depth integer NOT NULL,
  min_depth integer NOT NULL,
  seed text,
  success boolean NOT NULL,
  stop_reason text NOT NULL,
  combinations_used integer NOT NULL,
  max_combinations integer NOT NULL,
  user_id text REFERENCES users(id) ON DELETE SET NULL,
  raw_report jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_runs_model_depth_idx
  ON agent_runs (requested_depth, model, created_at);
