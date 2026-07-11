CREATE TABLE IF NOT EXISTS rate_limit_counters (
  scope text NOT NULL,
  key text NOT NULL,
  window_start timestamptz NOT NULL,
  count integer NOT NULL DEFAULT 0,
  PRIMARY KEY (scope, key, window_start)
);
