from __future__ import annotations

from pathlib import Path

from src.data.db import connect, repo_root


MIGRATIONS_DIR = repo_root() / "db" / "migrations"


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def ensure_migrations_table(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version text PRIMARY KEY,
          applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def applied_versions(connection) -> set[str]:
    rows = connection.execute("SELECT version FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def apply_migration(connection, path: Path) -> None:
    version = path.stem
    connection.execute(path.read_text(encoding="utf-8"))
    connection.execute(
        "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
        (version,),
    )


def main() -> None:
    with connect() as connection:
        ensure_migrations_table(connection)
        applied = applied_versions(connection)
        pending = [path for path in migration_files() if path.stem not in applied]
        for path in pending:
            apply_migration(connection, path)
            print(f"applied {path.name}")
        if not pending:
            print("database already up to date")


if __name__ == "__main__":
    main()
