from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg


DEFAULT_DATABASE_URL = "postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_database_url() -> str:
    env_path = repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect() -> psycopg.Connection:
    return psycopg.connect(load_database_url())
