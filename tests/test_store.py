def one(store, sql, *args):
    with store.conn.cursor() as cur:
        cur.execute(sql, args)
        return list(cur.fetchone().values())[0]


def test_planted_layer_is_loaded(seeded, smoke_corpus):
    c = smoke_corpus
    assert one(seeded, "SELECT count(*) FROM true_entities") == len(c.entities)
    assert one(seeded, "SELECT count(*) FROM true_mentions") == sum(len(x.true_mentions) for x in c.cards)
    assert one(seeded, "SELECT count(*) FROM true_facts") == sum(len(x.true_facts) for x in c.cards)


def test_no_forward_edge_reaches_the_database(seeded, smoke_corpus):
    assert one(seeded, "SELECT count(*) FROM edges e JOIN cards s ON s.id = e.src "
                       "JOIN cards d ON d.id = e.dst WHERE d.committed_at > s.committed_at") == 0
    assert one(seeded, "SELECT count(*) FROM edges e JOIN cards s ON s.id = e.src "
                       "WHERE e.valid_from <> s.committed_at") == 0
    assert "forward_edges_dropped" in smoke_corpus.stats


def test_extracted_tables_exist_empty(seeded):
    for t in ("entities", "mentions", "facts", "onto_classes", "extracted_cards"):
        assert one(seeded, f"SELECT count(*) FROM {t}") == 0


def test_reset_drops_the_new_tables(store, smoke_corpus):
    store.init(reset=True)
    assert one(store, "SELECT count(*) FROM true_mentions") == 0
    store.load_corpus(smoke_corpus)          # leave the session DB seeded for later tests
