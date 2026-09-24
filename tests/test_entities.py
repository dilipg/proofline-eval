import hashlib
import json
import re

import pytest

import proofline_eval as pe

GOLDEN = {"smoke": "49b2d1bc1a149fc2ca446eb103e50d4ff6c563bf80e97e1a9454108e27459bc6",
          "quick": "f7327cd74a111d10fb1ea7c18161f187656b6040fc25a1fdead7e12ed1fed3e7"}


def corpus_digest(corpus) -> str:
    cards = [(c.id, c.proofline_id, c.topic, c.kc_type, c.kc_stage, c.title, c.one_line, c.body,
              c.committed_at.isoformat(), c.version, c.supersedes_id, c.is_current,
              list(c.parent_ids), list(c.citation_ids), list(c.true_support))
             for c in corpus.cards]
    queries = {prov: [(q.id, q.text, q.as_of.isoformat(), q.source_card, sorted(q.rel.items()),
                       list(q.slices), q.answerable) for q in qs]
               for prov, qs in sorted(corpus.queries.items())}
    blob = json.dumps({"cards": cards, "queries": queries}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def build(profile, entities):
    cfg = pe.Config(profile=profile, entities=entities)
    return pe.build_corpus(cfg, cfg.p["cards"], cfg.p["prooflines"])


@pytest.fixture(scope="module")
def quick_on():
    return build("quick", True)


@pytest.mark.parametrize("profile", ["smoke", "quick"])
def test_no_entities_reproduces_todays_corpus(profile):
    c = build(profile, False)
    assert corpus_digest(c) == GOLDEN[profile]
    assert c.entities == [] and all(not x.true_mentions and not x.true_facts for x in c.cards)


def test_entity_layer_changes_only_bodies(quick_on):
    off = build("quick", False)
    n = len(pe.CONNECTIVE)
    assert [c.id for c in off.cards] == [c.id for c in quick_on.cards]
    for a, b in zip(off.cards, quick_on.cards):
        assert (a.title, a.one_line, a.committed_at, a.parent_ids, a.citation_ids, a.true_support) == \
               (b.title, b.one_line, b.committed_at, b.parent_ids, b.citation_ids, b.true_support)
        x, y = a.body.split(), b.body.split()
        assert y[:n] == x[:n] and y[len(y) - (len(x) - n):] == x[n:]
    q_off = {p: [(q.id, q.text, sorted(q.rel.items())) for q in qs] for p, qs in off.queries.items()}
    q_on = {p: [(q.id, q.text, sorted(q.rel.items())) for q in qs] for p, qs in quick_on.queries.items()}
    assert q_off == q_on


def test_every_planted_surface_is_in_its_card(quick_on):
    for c in quick_on.cards:
        for _eid, surface, role in c.true_mentions:
            assert surface in c.text and role in pe.MENTION_ROLES


def test_facts_respect_domain_and_range(quick_on):
    kind = {i: k for i, _n, k, _p in quick_on.entities}
    for c in quick_on.cards:
        for s, rel, o, v in c.true_facts:
            dom, rng = pe.ONTO_RELATIONS[rel]
            assert pe.top_class(kind[s]) in dom and pe.top_class(kind[o]) in rng
            assert (v is not None) == (rel == "reports")
            if v is not None:
                assert re.fullmatch(r"0\.\d\d", v)


def test_uses_come_after_introduction_in_one_chain(quick_on):
    by_id = {c.id: c for c in quick_on.cards}

    def root(c):
        while c.supersedes_id:
            c = by_id[c.supersedes_id]
        return c.id

    intro, users = {}, {}
    for c in quick_on.cards:
        for eid, _s, role in c.true_mentions:
            if role == "introduces":
                intro.setdefault(eid, set()).add(root(c))
            if role == "uses":
                users.setdefault(eid, []).append(c)
    assert all(len(r) == 1 for r in intro.values())
    for eid, cs in users.items():
        if eid in intro:
            t0 = by_id[next(iter(intro[eid]))].committed_at
            assert all(c.committed_at > t0 for c in cs)


def test_planted_names_never_collide_with_topic_vocab(quick_on):
    vocab = {w for ws in quick_on.topics.values() for w in ws}
    for _i, name, _k, _p in quick_on.entities:
        assert not set(re.findall(r"[a-z0-9]+", name.lower())) & vocab


def test_layer_has_every_feature_the_spec_needs(quick_on):
    s = quick_on.stats
    kinds = {k for _i, _n, k, _p in quick_on.entities}
    assert kinds == set(pe.ONTO_CLASSES) - {"Method", "Dataset"}
    for key in ("aliased_methods", "near_miss_methods", "bridges_linked", "bridges_unlinked",
                "value_changes"):
        assert s.get(key, 0) > 0, key     # the stats come from a Counter: zero means absent


def test_about_half_the_bridges_reuse_an_existing_citation(quick_on):
    # the spec's linked/unlinked split: P2 compares the DAG walk and the ontology on
    # both halves, so neither may be starved
    s = quick_on.stats
    share = s["bridges_linked"] / (s["bridges_linked"] + s["bridges_unlinked"])
    assert 0.35 <= share <= 0.65, share


def test_plant_is_deterministic():
    assert corpus_digest(build("smoke", True)) == corpus_digest(build("smoke", True))
