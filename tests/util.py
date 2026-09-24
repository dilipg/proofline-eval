"""Small in-memory fixtures shared by the P2 tests: an Index with hash embeddings and a
Query factory. Nothing here touches the database."""
from datetime import datetime, timedelta, timezone

import proofline_eval as pe

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
EMB = pe.HashEmbedder(64)


def day(n: float) -> datetime:
    return T0 + timedelta(days=n)


def index_from(spec):
    """spec: [(card_id, day, text, supersedes_id_or_None)] -> Index with hash embeddings."""
    superseded = {s for _c, _d, _t, s in spec if s}
    rows = {}
    for cid, d, text, sup in spec:
        v = EMB.encode([text])[0]
        rows[cid] = dict(id=cid, proofline_id="pl", txt=text,
                         embedding=v.astype("float32").tobytes(), ndim=EMB.dim,
                         is_current=cid not in superseded, supersedes_id=sup,
                         committed_at=day(d))
    return pe.Index(rows)


def query(text, as_of_day, **fields):
    base = dict(id="q", text=text, provenance="test", as_of=day(as_of_day), source_card=None,
                rel={}, slices=[], answerable=True)
    base.update(fields)
    return pe.Query(**base)
