import json
from types import SimpleNamespace

import proofline_eval as pe
from util import EMB, day, index_from, query

SPEC = [("A", 5, "quanibraion modrievment lattilency calibrated", None),
        ("B", 2, "thermoulaity caltroposis isostabity reached", None),
        ("N", 3, "gratilure retulaure noise", None)]
ENTS = {"x_m": ("Kavrel-7", "SamplingMethod")}
MENTS = [("A", "x_m", "Kavrel-7", "uses"), ("B", "x_m", "Kavrel-7", "mentions")]


def _scorer(walker):
    idx = index_from(SPEC)
    idx.onto = pe.OntologyIndex(idx, ENTS, MENTS, [])
    sc = pe.OntoLlmWalk()
    sc.prepare(idx, EMB)
    sc.walker = walker
    return sc


def test_mock_walker_follows_the_bridge_offline():
    out = _scorer(pe.MockWalker()).run(query("quanibraion modrievment lattilency", 6), 3)
    assert out.ids[0] == "A" and "B" in out.ids and out.chain


class _FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def create(self, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        text, stop = r
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def test_anthropic_walker_filters_and_counts_bad_replies(tmp_path):
    msgs = _FakeMessages([
        (json.dumps({"next": ["B", "GHOST"], "done": False, "order": ["B", "GHOST", "A"]}), "end_turn"),
        ("not json", "end_turn"),
        ("", "refusal"),
    ])
    w = pe.AnthropicWalker("claude-haiku-4-5", client=SimpleNamespace(messages=msgs),
                           cache_path=tmp_path / "llm.sqlite")
    ev = [{"id": "A", "date": "2025-01-06", "text": "a"}]
    cands = [{"id": "B", "date": "2025-01-03", "text": "b"}]
    r1 = w.choose("q1", day(6), ev, cands, 0)
    assert r1["next"] == ["B"] and r1["order"] == ["B", "A"] and not r1["done"]
    r2 = w.choose("q2", day(6), ev, cands, 0)
    r3 = w.choose("q3", day(6), ev, cands, 0)
    assert r2["done"] and r3["done"] and w.errors == 2
    again = w.choose("q1", day(6), ev, cands, 0)              # served from the cache
    assert again == r1 and msgs.calls == 3


def test_make_walker_parses_the_spec():
    assert isinstance(pe.make_walker("mock"), pe.MockWalker)
    assert pe.make_walker("anthropic:claude-haiku-4-5").model == "claude-haiku-4-5"


def test_anthropic_walker_needs_a_key_from_the_environment(monkeypatch):
    # the SDK only fails at request time; a walker that never ran must not print a verdict
    import pytest
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        pe.make_walker("anthropic:claude-haiku-4-5")


def test_cache_key_covers_the_system_prompt_and_schema(tmp_path, monkeypatch):
    w = pe.AnthropicWalker("claude-haiku-4-5", client=SimpleNamespace(messages=None),
                           cache_path=tmp_path / "llm.sqlite")
    k = w.cache_key("same prompt")
    monkeypatch.setattr(pe, "WALK_SYSTEM", pe.WALK_SYSTEM + " (edited)")
    assert w.cache_key("same prompt") != k


def test_a_non_object_reply_is_an_error_and_is_never_cached(tmp_path):
    ok = json.dumps({"next": [], "done": True, "order": []})
    msgs = _FakeMessages([("[]", "end_turn"), (ok, "end_turn")])
    w = pe.AnthropicWalker("claude-haiku-4-5", client=SimpleNamespace(messages=msgs),
                           cache_path=tmp_path / "llm.sqlite")
    ev = [{"id": "A", "date": "2025-01-06", "text": "a"}]
    assert w.choose("q", day(6), ev, [], 0)["done"] and w.errors == 1
    w.choose("q", day(6), ev, [], 0)
    assert msgs.calls == 2                       # the bad reply was not served from cache


def test_a_bad_value_already_in_the_cache_is_asked_again(tmp_path):
    ok = json.dumps({"next": [], "done": True, "order": ["A"]})
    msgs = _FakeMessages([(ok, "end_turn")])
    w = pe.AnthropicWalker("claude-haiku-4-5", client=SimpleNamespace(messages=msgs),
                           cache_path=tmp_path / "llm.sqlite")
    ev = [{"id": "A", "date": "2025-01-06", "text": "a"}]
    real_get = w.cache.get
    # a value cached by a build that did not validate replies
    w.cache.get = lambda k: w.cache.put(k, ["stale"]) or real_get(k)
    assert w.choose("q", day(6), ev, [], 0)["order"] == ["A"]
    assert msgs.calls == 1 and w.errors == 0
