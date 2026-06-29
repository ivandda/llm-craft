# Postgres Dataset DB

The local Postgres service stores the curated `final-10k` recipe dataset and the
web app state for users, sessions, profiles, and leaderboards.

## Startup

From the repository root:

```bash
docker compose up -d postgres
uv run python -m src.data.db_migrate
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```

Default connection string:

```text
postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft
```

Set `DATABASE_URL` in `.env` or `apps/web/.env.local` when overriding defaults.

## Persistence

Postgres data is bind-mounted to:

```text
var/postgres-data/
```

That directory is ignored by git through the existing `var/` ignore rule. It
survives container restarts, rebuilds, and `docker compose down`. It is deleted
only if the directory is removed manually.

## Schema

The migration creates:

- Dataset tables: `dataset_imports`, `recipe_pairs`, `recipe_candidates`,
  `dataset_rejections`, and `dataset_manifests`.
- Web app tables: `users`, `sessions`, `user_profiles`,
  `featured_achievements`, and `leaderboard_entries`.

The importer stores normalized relational fields and preserves each source row
in `jsonb` columns (`raw_record`, `raw_candidate`, or `raw_manifest`).

## Web Behavior

`POST /api/combine` first looks up the normalized input pair in `final-10k`.
When the pair exists, rank 1 is returned as the main result and all ranked
candidates are returned as `knownOutputs`. Unknown pairs keep the deterministic
mock fallback and are not written to the database.

The seeded local user remains:

```text
username: admin
password: admin
```

## Backup And Reset

Backup:

```bash
docker compose exec postgres pg_dump -U llm_craft -d llm_craft > backup.sql
```

Reset all local database data:

```bash
docker compose down
Remove-Item -Recurse -Force var/postgres-data
docker compose up -d postgres
uv run python -m src.data.db_migrate
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```

Refreshing only the dataset is safe for web app users and leaderboards:

```bash
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```
