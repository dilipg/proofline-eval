import proofline_eval as pe


def test_multihop_round_trips(seeded, smoke_corpus):
    want = {q.id: q for q in smoke_corpus.queries["multihop"]}
    # storage, not filtering: include what the disconnection filter (another test) dropped
    got = {q.id: q for q in pe.load_queries(seeded, "multihop", include_dropped=True)}
    assert set(got) == set(want) and want
    for qid, q in want.items():
        g = got[qid]
        assert (g.intent, g.family, g.pair_id, g.answerable) == (q.intent, q.family, q.pair_id, q.answerable)
        assert g.slots == q.slots and g.answer == q.answer and g.rel == q.rel


def test_dropped_queries_are_excluded_unless_asked(seeded):
    qid = pe.load_queries(seeded, "multihop")[0].id
    with seeded.conn.cursor() as cur:
        cur.execute("UPDATE queries SET dropped = 'single_hop' WHERE id = %s", (qid,))
    try:
        assert qid not in {q.id for q in pe.load_queries(seeded, "multihop")}
        assert qid in {q.id for q in pe.load_queries(seeded, "multihop", include_dropped=True)}
    finally:
        with seeded.conn.cursor() as cur:
            cur.execute("UPDATE queries SET dropped = NULL WHERE id = %s", (qid,))


def test_older_provenances_load_unchanged(seeded, smoke_corpus):
    q = pe.load_queries(seeded, "planted")[0]
    assert q.slots == [] and q.intent == "current" and q.family is None
