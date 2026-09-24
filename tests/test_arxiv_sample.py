import proofline_eval as pe


def test_snowball_does_not_depend_on_iteration_order():
    # adj holds sets of arXiv ids, and str hashing is salted per process: a snowball
    # that follows set order draws a different corpus on every seed (seen: 33,675
    # papers one run, 33,657 the next, from the same data and --seed)
    nbrs = {"a": ["b", "c", "d"], "b": ["e", "f"], "c": ["g", "h"], "d": ["i"]}
    fwd = {p: list(qs) for p, qs in nbrs.items()}
    rev = {p: list(reversed(qs)) for p, qs in nbrs.items()}
    nver = dict.fromkeys("abcdefghi", 1)
    for target in (3, 5, 7):
        assert pe._snowball(fwd, ["a"], target, nver) == pe._snowball(rev, ["a"], target, nver)


FIXTURE = '''
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, {root!r})
import proofline_eval as pe
pe.DATA_DIR = Path({data!r})
cfg = pe.Config(profile="smoke", source="arxiv", max_eval_queries=12)
c = pe.build_corpus_arxiv(cfg, 60)
rows = sorted((p, q.id, q.text, q.source_card, sorted(q.rel)) for p, ql in c.queries.items() for q in ql)
print("DIGEST", hashlib.sha256(json.dumps([sorted(x.id for x in c.cards), rows]).encode()).hexdigest())
'''


def test_arxiv_corpus_is_the_same_under_any_hash_seed(tmp_path):
    # the card list followed a set's order, so every rng.sample over it -- the eval
    # queries, the generated query words -- changed with the process's hash salt
    import json, os, subprocess, sys
    from pathlib import Path
    words = "sparse attention kernels scale linearly with sequence length under mild assumptions".split()
    with (tmp_path / "arxiv-subset.jsonl").open("w", encoding="utf-8") as f:
        for i in range(40):
            f.write(json.dumps(dict(
                id=f"2101.{i:05d}", title=f"Paper number {i} on attention", categories="cs.LG",
                abstract=" ".join(words[(i + j) % len(words)] for j in range(40)),
                versions=[dict(version=f"v{v}", created=f"Mon, {1 + i % 27:d} Feb 2021 1{v}:00:00 GMT")
                          for v in range(1, 2 + i % 3)])) + "\n")
    with (tmp_path / "citations-arxiv.tsv").open("w") as f:
        for i in range(1, 40):
            for j in {0, i // 2, i // 3}:
                f.write(f"2101.{i:05d}\t2101.{j:05d}\n")
    root = str(Path(pe.__file__).resolve().parent)
    code = FIXTURE.format(root=root, data=str(tmp_path))
    digests = set()
    for seed in ("1", "2", "3"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"})
        assert r.returncode == 0, r.stderr[-800:]
        digests.add(next(l for l in r.stdout.splitlines() if l.startswith("DIGEST")))
    assert len(digests) == 1
