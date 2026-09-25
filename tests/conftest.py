"""Test setup. The state dir is fixed BEFORE the harness is imported, so the embedded
Postgres lives in .proofline/test-state (gitignored, reused across runs) and never in
the real .proofline/pgdata a user's corpus lives in."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ["PROOFLINE_STATE"] = str(ROOT / ".proofline" / "test-state")

import proofline_eval as pe  # noqa: E402

# after the import: load_dotenv() may have set it from .env
os.environ.pop("DATABASE_URL", None)


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def store():
    s = pe.Store(pe.Config(profile="smoke"))
    yield s
    s.close()


@pytest.fixture(scope="session")
def smoke_corpus():
    cfg = pe.Config(profile="smoke")
    return pe.build_corpus(cfg, cfg.p["cards"], cfg.p["prooflines"])


@pytest.fixture(scope="session")
def seeded(store, smoke_corpus):
    store.init(reset=True)
    store.load_corpus(smoke_corpus)
    return store


@pytest.fixture
def dag_db(store):
    """A scratch schema with only `cards` and `edges`, on its own connection, so the SQL
    DAG walk can be tested on a hand-built record without touching the session store."""
    import psycopg
    from psycopg.rows import dict_row
    conn = psycopg.connect(store.url, row_factory=dict_row, autocommit=True)
    conn.execute("DROP SCHEMA IF EXISTS dagtest CASCADE")
    conn.execute("CREATE SCHEMA dagtest")
    conn.execute("SET search_path TO dagtest")
    conn.execute("CREATE TABLE cards (id text PRIMARY KEY, committed_at timestamptz NOT NULL, "
                 "supersedes_id text)")
    conn.execute("CREATE TABLE edges (src text NOT NULL, dst text NOT NULL, kind text NOT NULL, "
                 "valid_from timestamptz NOT NULL)")
    yield conn
    conn.execute("DROP SCHEMA dagtest CASCADE")
    conn.close()
