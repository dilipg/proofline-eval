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


def flood(spec, n=12):
    """Add n filler cards that match the question's first word, so the first stage's top
    ten is all fillers and a card that shares no word with the question can only enter
    the top k through the walk. With three or four cards, dense retrieval scores every
    card and a walk test passes with no walk at all."""
    return spec + [(f"N{i:02d}", 3, f"quanibraion filler{i} padding{i}", None) for i in range(n)]


def load_dag(conn, spec, links):
    """spec as for index_from; links: [(src, dst, day)] citations, valid from `day`."""
    with conn.cursor() as cur:
        for cid, d, _text, sup in spec:
            cur.execute("INSERT INTO cards VALUES (%s, %s, %s)", (cid, day(d), sup))
        for s, t, d in links:
            cur.execute("INSERT INTO edges VALUES (%s, %s, 'citation', %s)", (s, t, day(d)))
