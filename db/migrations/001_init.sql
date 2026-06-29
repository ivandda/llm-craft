CREATE TABLE IF NOT EXISTS schema_migrations (
  version text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dataset_imports (
  dataset_name text PRIMARY KEY,
  source_dir text NOT NULL,
  imported_at timestamptz NOT NULL DEFAULT now(),
  train_count integer NOT NULL DEFAULT 0,
  dev_count integer NOT NULL DEFAULT 0,
  test_count integer NOT NULL DEFAULT 0,
  rejected_count integer NOT NULL DEFAULT 0,
  raw_metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS recipe_pairs (
  pair_id text PRIMARY KEY,
  dataset_name text NOT NULL REFERENCES dataset_imports(dataset_name) ON DELETE CASCADE,
  split text NOT NULL CHECK (split IN ('train', 'dev', 'test')),
  input_a text NOT NULL,
  input_b text NOT NULL,
  raw_record jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (dataset_name, input_a, input_b)
);

CREATE INDEX IF NOT EXISTS recipe_pairs_lookup_idx
  ON recipe_pairs (dataset_name, input_a, input_b);

CREATE TABLE IF NOT EXISTS recipe_candidates (
  candidate_id text PRIMARY KEY,
  pair_id text NOT NULL REFERENCES recipe_pairs(pair_id) ON DELETE CASCADE,
  output text NOT NULL,
  source text NOT NULL CHECK (source IN ('observed', 'teacher')),
  rationale text,
  rank integer NOT NULL CHECK (rank > 0),
  raw_candidate jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (pair_id, rank),
  UNIQUE (pair_id, output)
);

CREATE INDEX IF NOT EXISTS recipe_candidates_pair_rank_idx
  ON recipe_candidates (pair_id, rank);

CREATE TABLE IF NOT EXISTS dataset_rejections (
  rejection_id text PRIMARY KEY,
  dataset_name text NOT NULL REFERENCES dataset_imports(dataset_name) ON DELETE CASCADE,
  split text NOT NULL,
  input_a text NOT NULL,
  input_b text NOT NULL,
  outputs jsonb NOT NULL,
  reject_reason text NOT NULL,
  detail text,
  raw_record jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dataset_rejections_dataset_split_idx
  ON dataset_rejections (dataset_name, split);

CREATE TABLE IF NOT EXISTS dataset_manifests (
  manifest_id text PRIMARY KEY,
  dataset_name text NOT NULL REFERENCES dataset_imports(dataset_name) ON DELETE CASCADE,
  manifest_name text NOT NULL,
  raw_manifest jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (dataset_name, manifest_name)
);

CREATE TABLE IF NOT EXISTS users (
  id text PRIMARY KEY,
  username text NOT NULL UNIQUE,
  display_name text NOT NULL,
  password_hash text NOT NULL,
  password_salt text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
  id text PRIMARY KEY,
  user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sessions_user_id_idx
  ON sessions (user_id);

CREATE TABLE IF NOT EXISTS user_profiles (
  user_id text PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  display_name text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS featured_achievements (
  user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  position integer NOT NULL CHECK (position >= 0),
  element_id text NOT NULL,
  name text NOT NULL,
  emoji text,
  featured_at timestamptz NOT NULL,
  PRIMARY KEY (user_id, position),
  UNIQUE (user_id, element_id)
);

CREATE TABLE IF NOT EXISTS leaderboard_entries (
  id text PRIMARY KEY,
  goal_id text NOT NULL,
  goal_title text NOT NULL,
  user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  username text NOT NULL,
  display_name text NOT NULL,
  combinations_used integer NOT NULL CHECK (combinations_used > 0),
  completed_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (goal_id, user_id)
);

CREATE INDEX IF NOT EXISTS leaderboard_entries_goal_rank_idx
  ON leaderboard_entries (goal_id, combinations_used, completed_at);
