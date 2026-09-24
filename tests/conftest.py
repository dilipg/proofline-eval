"""Test setup. The state dir is fixed BEFORE the harness is imported, so the embedded
Postgres lives in .proofline/test-state (gitignored, reused across runs) and never in
the real .proofline/pgdata a user's corpus lives in."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
