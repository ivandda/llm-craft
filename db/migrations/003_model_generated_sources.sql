ALTER TABLE dpo_preference_events
  DROP CONSTRAINT IF EXISTS dpo_preference_events_source_check;

ALTER TABLE dpo_preference_events
  ADD CONSTRAINT dpo_preference_events_source_check
  CHECK (source IN ('known_recipe', 'mock_model', 'model_generated'));
