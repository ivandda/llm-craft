CREATE TABLE IF NOT EXISTS dpo_preference_events (
  id text PRIMARY KEY,
  user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  mode text NOT NULL CHECK (mode IN ('sandbox', 'goal')),
  goal_id text,
  input_a jsonb NOT NULL,
  input_b jsonb NOT NULL,
  shown_outputs jsonb NOT NULL,
  selected_output jsonb NOT NULL,
  rejected_outputs jsonb NOT NULL,
  inventory_snapshot jsonb NOT NULL,
  combination_index integer NOT NULL CHECK (combination_index > 0),
  source text NOT NULL CHECK (source IN ('known_recipe', 'mock_model')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dpo_preference_events_user_created_idx
  ON dpo_preference_events (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS dpo_preference_events_goal_created_idx
  ON dpo_preference_events (goal_id, created_at DESC)
  WHERE goal_id IS NOT NULL;
