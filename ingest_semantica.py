# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.26", "psycopg[binary]>=3.1",
#   "pgserver>=0.1.4; sys_platform == 'darwin' or platform_machine == 'x86_64' or platform_machine == 'AMD64'",
#   "semantica[llm-anthropic,viz]==0.7.0",
# ]
# ///
"""Ontology ingestion with Semantica, and pictures of what was ingested.

Extract typed entities and relations from every card, resolve them into corpus-level
entities, give every fact a validity window from the supersession chain, and write it
all back next to the cards. With --viz, draw the DAG, the ontology graph, the class
hierarchy and a timeline, and export graph.json for `semantica-explorer`.

Semantica does the reading, the merging, the ontology and the drawing. It never runs at
query time: the harness walks these tables through the same as-of mask as everything
else, because a per-query snapshot rebuild is the 2-minute bug (invariant 4).

    uv run ingest_semantica.py --method regex              # offline, no key, all cards
    uv run ingest_semantica.py --method llm                # anthropic:claude-haiku-4-5, 300 cards
    uv run ingest_semantica.py --method llm --llm anthropic:claude-opus-5 --extract-limit 0
    uv run ingest_semantica.py --viz-only --method regex   # draw an existing extraction

Every Semantica call goes through SemanticaAdapter, so an API change in a weekly release
touches one class. Everything above it is pure, and is what tests/ exercises.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofline_eval as pe  # noqa: E402

# The ontology the extractor is asked to fill: the DESIGN, never the planted instances.
LEAF_TYPES = [k for k, parent in pe.ONTO_CLASSES.items()
              if parent is not None or k not in ("Method", "Dataset")]
ENTITY_RELATIONS = list(pe.ONTO_RELATIONS)
# The offline extractor. These patterns are written against the synthetic naming scheme,
# so its entity numbers are an upper bound for a regex, not evidence about one. Each is
# wrapped in (?-i:...) because Semantica's regex method forces re.IGNORECASE, under which
# the Method pattern reads "scores 0" out of "scores 0.74".
REGEX_PATTERNS = {k: f"(?-i:{v})" for k, v in {
    "Method": r"\b(?:[A-Z][a-z]+[- ]\d+|[A-Z]{2}-\d+)\b",
    "Dataset": r"\b[A-Z][a-z]+-(?:QA|Bench|Corpus|Text)\b",
    "Metric": r"\b[a-z]+-score\b",
    "Material": r"\b[a-z]+ (?:oxide|nitride|alloy|polymer)\b",
    "Organism": r"\b[A-Z][a-z]+ [a-z]+ii\b",
}.items()}
# Card -> entity roles from cue phrases. The cues are the planted templates' own words,
# so B1's role accuracy is a plumbing check and is labelled as one there.
ROLE_CUES = (
    ("introduces", re.compile(r"\b(?:we introduce|presents|proposed in this note|we release|"
                              r"assembled for this study|is introduced here|we synthesise|"
                              r"prepared here|preparation of)\b")),
    ("studies", re.compile(r"\b(?:model organism|we study|samples of)\b")),
    ("uses", re.compile(r"\b(?:we adopt|builds on|relies on)\b")),
)
# Part of every cache key. Bump it whenever SemanticaAdapter.extract's output can change
# for the same text (a pattern, a filter, an extractor argument), or the cache will keep
# answering with what the old code produced.
ADAPTER_VERSION = "2"
PROBE = ("we introduce Kavrel-7 , a sampling method , and evaluate it on Orvane-QA , where "
         "it reaches 0.74 vex-score . Kavrel-7 extends Dorvin-2 .")
_CLASS_TAIL = re.compile(r"\s+(?:benchmark|corpus|dataset|method|approach|material)$", re.I)


@dataclass
class CardExtraction:
    card_id: str
    entities: list            # [(surface, label)]
    triplets: list            # [(subject surface, predicate, object surface)]
    error: Optional[str] = None


# ---- pure helpers -----------------------------------------------------------------

def source_id(method: str, llm: Optional[str]) -> str:
    return "semantica:regex" if method == "regex" else f"semantica:llm:{llm}"


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def entity_key(surface: str) -> str:
    """Case, spacing and punctuation fold together ('Kavrel-7' == 'Kavrel 7'). A trailing
    class word is the extractor describing the thing, not naming it ('Orvane-QA benchmark')."""
    return re.sub(r"[^a-z0-9]+", "", _CLASS_TAIL.sub("", surface.strip()).lower())


def parse_value(s: str) -> Optional[str]:
    """A reported value is a leading decimal ('0.74 vex-score'). A name with a digit in it
    ('Kavrel-7') is not one."""
    m = re.match(r"\s*(\d*\.\d+)(?![\d.])", s)
    return f"{float(m.group(1)):.2f}" if m else None


def snake(pred: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pred.strip()).lower().replace(" ", "_")


def onto_kind(label: str) -> str:
    flat = re.sub(r"[^a-z]", "", label.lower())
    return next((k for k in pe.ONTO_CLASSES if k.lower() == flat), label)


def infer_role(sentence: str) -> str:
    s = sentence.lower()
    for role, rx in ROLE_CUES:
        if rx.search(s):
            return role
    return "mentions"


def strongest(a: str, b: str) -> str:
    return min(a, b, key=pe.MENTION_ROLES.index)


def normalize_card(x: CardExtraction, text: str) -> tuple[list[tuple], list[tuple]]:
    """One card's raw extraction as (key, surface, label, role) mentions and
    (subj_key, rel, obj_key, value) facts. A value-bearing object ('0.74 vex-score') on
    measured_by/reports becomes a `reports` fact when the card evaluates that method on
    exactly one dataset. With several datasets the value's referent is ambiguous, so it
    is dropped rather than guessed."""
    sentences = [s for s in re.split(r"\s\.(?:\s|$)", text) if s.strip()]
    ments: dict[tuple[str, str], tuple[str, str]] = {}
    for surface, label in x.entities:
        k = entity_key(surface)
        if not k:
            continue
        role = "mentions"
        for s in sentences:
            if surface in s:
                role = strongest(role, infer_role(s))
        prev = ments.get((k, surface))
        ments[(k, surface)] = (label, strongest(prev[1], role) if prev else role)
    known = {k for k, _s in ments}
    facts: set[tuple] = set()
    values: dict[str, list[str]] = defaultdict(list)
    evaluated: dict[str, set[str]] = defaultdict(set)
    for s, p, o in x.triplets:
        rel, sk, ok = snake(p), entity_key(s), entity_key(o)
        if sk not in known:
            continue
        if rel in ("measured_by", "reports") and ok not in known:
            v = parse_value(o)
            if v:
                values[sk].append(v)
            continue
        if ok not in known:
            continue
        if rel == "evaluated_on":
            evaluated[sk].add(ok)
        facts.add((sk, rel, ok, None))
    for sk, vs in values.items():
        if len(evaluated[sk]) == 1:
            (dk,) = evaluated[sk]
            facts.update((sk, "reports", dk, v) for v in vs)
    return ([(k, s, label, role) for (k, s), (label, role) in sorted(ments.items())],
            sorted(facts, key=repr))


def resolve_entities(per_card: dict[str, list[tuple]],
                     groups_for: Callable[[list[dict]], list[list[str]]]
                     ) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """Merge per-card entity keys into corpus-level entities. `groups_for` gets the
    candidates of ONE top-level class ({id, name, type}) and returns groups of ids to
    merge: Semantica's DuplicateDetector in a run, a stub in tests. Merging never
    crosses classes, so a method and a dataset that share a stem stay apart."""
    surf: dict[str, Counter] = defaultdict(Counter)
    lab: dict[str, Counter] = defaultdict(Counter)
    for rows in per_card.values():
        for key, s, label, _role in rows:
            surf[key][s] += 1
            lab[key][onto_kind(label)] += 1
    parent = {k: k for k in surf}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    # String similarity cannot tell siblings apart: on Semantica 0.7.0 distinct names score
    # HIGHER than true aliases (Brakar-Corpus~Brakar-Text 0.80, Dorula-78~Dorith-85 0.78,
    # alias Kavrel-7~KA-7 0.70). A number in a name is identity, so the detector is only
    # offered names of one class that carry the SAME number; names without one merge on
    # exact key alone.
    groups: dict[tuple, list[str]] = defaultdict(list)
    for k in sorted(surf):
        kind = lab[k].most_common(1)[0][0]
        top = pe.top_class(kind) if kind in pe.ONTO_CLASSES else kind
        digits = tuple(re.findall(r"\d+", surf[k].most_common(1)[0][0]))
        if digits:
            groups[(top, digits)].append(k)
    for (top, _digits), keys in sorted(groups.items()):
        if len(keys) < 2:
            continue
        cands = [{"id": k, "name": surf[k].most_common(1)[0][0], "type": top} for k in keys]
        for g in groups_for(cands):
            g = [k for k in g if k in keys]
            for k in g[1:]:
                parent[find(k)] = find(g[0])
    members: dict[str, list[str]] = defaultdict(list)
    for k in sorted(surf):
        members[find(k)].append(k)
    rows, key2id = [], {}
    for root, ks in sorted(members.items()):
        names, kinds = Counter(), Counter()
        for k in ks:
            names.update(surf[k])
            kinds.update(lab[k])
        eid = f"x_{root}"
        rows.append((eid, names.most_common(1)[0][0], kinds.most_common(1)[0][0]))
        for k in ks:
            key2id[k] = eid
    return rows, key2id


def validity_windows(rows: Sequence[dict]) -> dict[str, tuple[datetime, Optional[datetime]]]:
    """A card's facts hold from its commit until its successor's commit (None: still)."""
    succ = {r["supersedes_id"]: r["id"] for r in rows if r["supersedes_id"]}
    ts = {r["id"]: r["committed_at"] for r in rows}
    return {cid: (ts[cid], ts.get(succ.get(cid))) for cid in ts}


def assemble(results: dict[str, tuple[list, list]], key2id: dict[str, str],
             windows: dict[str, tuple[datetime, Optional[datetime]]]
             ) -> tuple[list[tuple], list[tuple]]:
    mentions, facts = set(), set()
    for cid, (ments, fs) in results.items():
        for k, surface, _label, role in ments:
            mentions.add((cid, key2id[k], surface, role))
        vf, vu = windows[cid]
        for sk, rel, ok, v in fs:
            facts.add((cid, key2id[sk], rel, key2id[ok], v, vf, vu))
    return sorted(mentions), sorted(facts, key=repr)


def pick_cards(rows: list[dict], limit: int, priority: set[str], seed: int) -> list[dict]:
    if not limit or limit >= len(rows):
        return rows
    first = [r for r in rows if r["id"] in priority]
    rest = [r for r in rows if r["id"] not in priority]
    random.Random(seed).shuffle(rest)
    return (first + rest)[:limit]


def error_gate(n_errors: int, n_total: int, limit: float = 0.05) -> None:
    if n_total and n_errors / n_total > limit:
        raise SystemExit(f"{n_errors}/{n_total} cards failed extraction (> {limit:.0%}); nothing "
                         "written. A partial extraction would be graded as low recall, which it is not.")


class ExtractCache:
    """(extractor id, card text) -> raw extraction. Lives outside the database, so a
    re-seed does not re-bill a model for text it has already read."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS x (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self.db.commit()
        self.lock = threading.Lock()

    @staticmethod
    def key(source: str, text: str) -> str:
        return hashlib.sha256(f"{ADAPTER_VERSION}\x00{source}\x00{text}".encode("utf-8")).hexdigest()

    def get(self, k: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT v FROM x WHERE k = ?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k: str, v: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO x VALUES (?, ?)", (k, json.dumps(v)))
            self.db.commit()


def write_back(store: pe.Store, source: str, card_ids: Sequence[str], entities: Sequence[tuple],
               mentions: Sequence[tuple], facts: Sequence[tuple], classes: Sequence[tuple]) -> None:
    """Replace everything `source` wrote before, in one transaction. A re-run is a
    replacement, never an append, and a failed run leaves the previous one intact."""
    with store.conn.transaction(), store.conn.cursor() as cur:
        for t in ("entities", "mentions", "facts", "onto_classes", "extracted_cards"):
            cur.execute(f"DELETE FROM {t} WHERE source = %s", (source,))
        with cur.copy("COPY extracted_cards (card_id, source) FROM STDIN") as cp:
            for cid in card_ids:
                cp.write_row((cid, source))
        with cur.copy("COPY entities (id, name, kind, source) FROM STDIN") as cp:
            for e in entities:
                cp.write_row((*e, source))
        with cur.copy("COPY mentions (card_id, entity_id, surface, role, source) FROM STDIN") as cp:
            for m in mentions:
                cp.write_row((*m, source))
        with cur.copy("COPY facts (card_id, subj, rel, obj, value, source, valid_from, valid_until) "
                      "FROM STDIN") as cp:
            for cid, s, r, o, v, vf, vu in facts:
                cp.write_row((cid, s, r, o, v, source, vf, vu))
        with cur.copy("COPY onto_classes (source, class, parent) FROM STDIN") as cp:
            for c, p in classes:
                cp.write_row((source, c, p))


def fetch_cards(store: pe.Store) -> list[dict]:
    with store.conn.cursor() as cur:
        cur.execute("SELECT id, proofline_id, title, txt, committed_at, supersedes_id, is_current "
                    "FROM cards ORDER BY id")
        return cur.fetchall()


def priority_cards(store: pe.Store) -> set[str]:
    """Cards a multi-hop query needs (P2's 'multihop' provenance). Empty until P2 exists."""
    with store.conn.cursor() as cur:
        cur.execute("SELECT DISTINCT r.card_id FROM qrels r JOIN queries q ON q.id = r.query_id "
                    "WHERE q.provenance = 'multihop'")
        return {r["card_id"] for r in cur.fetchall()}


# ---- the only place Semantica is called ------------------------------------------

class SemanticaAdapter:
    def __init__(self, method: str, llm: Optional[str]):
        self.method, self.llm = method, llm
        self.provider, self.model = (llm.split(":", 1) if llm else (None, None))
        self._local = threading.local()

    def _tools(self):
        t = getattr(self._local, "tools", None)
        if t is None:
            from semantica.semantic_extract import (NamedEntityRecognizer, RelationExtractor,
                                                    TripletExtractor)
            if self.method == "llm":
                t = (NamedEntityRecognizer(methods=["llm"], provider=self.provider,
                                           llm_model=self.model, entity_types=LEAF_TYPES),
                     TripletExtractor(method="llm", provider=self.provider, llm_model=self.model,
                                      include_temporal=True, triplet_types=ENTITY_RELATIONS))
            else:
                t = (NamedEntityRecognizer(methods=["regex"], patterns=REGEX_PATTERNS),
                     RelationExtractor(method="cooccurrence", relation_types=ENTITY_RELATIONS))
            self._local.tools = t
        return t

    def extract(self, card_id: str, text: str) -> CardExtraction:
        ner, rel = self._tools()
        try:
            if self.method == "llm":
                ents = ner.extract_entities(text, entity_types=LEAF_TYPES, silent_fail=False)
                trips = rel.extract_triplets(text, entities=ents, silent_fail=False)
                triplets = [(t.subject, t.predicate, t.object) for t in trips]
            else:
                ents = ner.extract_entities(text, patterns=REGEX_PATTERNS)
                rels = rel.extract_relations(text, ents)
                triplets = [(r.subject.text, r.predicate, r.object.text) for r in rels]
            # When the requested method finds nothing, Semantica silently falls back to its
            # built-in pattern NER (Title Case card titles become PERSON). Those entities
            # were never asked for; keep only what the chosen method produced.
            ents = [e for e in ents if (e.metadata or {}).get("extraction_method") != "pattern"]
            return CardExtraction(card_id, [(e.text, e.label) for e in ents], triplets)
        except Exception as e:  # one card's failure is counted, not fatal; see error_gate
            return CardExtraction(card_id, [], [], error=f"{type(e).__name__}: {str(e)[:200]}")

    def preflight(self) -> None:
        """silent_fail=False surfaces errors per card, but an extractor that returns nothing
        without raising would still write an empty extraction that B1 reads as recall 0.
        One known sentence must come back non-empty before any card is read."""
        x = self.extract("probe", PROBE)
        if x.error or not x.entities or (self.method == "llm" and not x.triplets):
            raise SystemExit(f"preflight failed for {source_id(self.method, self.llm)}: "
                             f"{x.error or 'empty extraction'}")
        pe.log(f"  preflight ok: {len(x.entities)} entities, {len(x.triplets)} relations from the probe")

    def duplicate_groups(self, cands: list[dict]) -> list[list[str]]:
        from semantica.deduplication import DuplicateDetector
        groups = DuplicateDetector(similarity_threshold=0.7).detect_duplicate_groups(cands)
        return [[e["id"] for e in g.entities] for g in groups]

    def ontology(self, entities: Sequence[tuple], facts: Sequence[tuple]) -> dict:
        from semantica.ontology import OntologyGenerator
        rels = sorted({(s, o, r) for _c, s, r, o, *_ in facts})
        return OntologyGenerator(base_uri="https://proofline.local/onto/").generate_ontology(
            {"entities": [{"id": i, "type": k, "name": n} for i, n, k in entities],
             "relationships": [{"source": s, "target": o, "type": r} for s, o, r in rels]},
            min_occurrences=2)


# ---- the run ---------------------------------------------------------------------

def run_extraction(store: pe.Store, rows: list[dict], a: argparse.Namespace, source: str) -> None:
    llm = a.llm if a.method == "llm" else None
    limit = a.extract_limit if a.extract_limit is not None else (300 if a.method == "llm" else 0)
    adapter = SemanticaAdapter(a.method, llm)
    adapter.preflight()
    cache = ExtractCache(pe.STATE_DIR / "cache" / "extract.sqlite")
    todo = pick_cards(rows, limit, priority_cards(store), seed=20260924)
    workers = max(1, a.workers) if a.method == "llm" else 1
    pe.log(f"extracting {len(todo)} of {len(rows)} cards with {source}"
           + (f"  ({workers} workers, 2 LLM calls per uncached card)" if llm else ""))
    t0 = time.time()

    def one(r: dict) -> tuple[CardExtraction, bool]:
        k = cache.key(source, r["txt"])
        got = cache.get(k)
        if got is not None:
            return CardExtraction(r["id"], [tuple(e) for e in got["e"]],
                                  [tuple(t) for t in got["t"]]), True
        x = adapter.extract(r["id"], r["txt"])
        if x.error is None:
            cache.put(k, {"e": x.entities, "t": x.triplets})
        return x, False

    done: list[CardExtraction] = []
    hits = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (x, cached) in enumerate(ex.map(one, todo), 1):
            done.append(x)
            hits += cached
            if i % 50 == 0 or i == len(todo):
                pe.log(f"  {i}/{len(todo)}  cached={hits}  "
                       f"errors={sum(1 for d in done if d.error)}  {time.time() - t0:.0f}s")
    errors = [d for d in done if d.error]
    for d in errors[:5]:
        pe.log(f"  ERROR {d.card_id}: {d.error}")
    error_gate(len(errors), len(done))
    text_of = {r["id"]: r["txt"] for r in rows}
    ok = [d for d in done if d.error is None]
    results = {d.card_id: normalize_card(d, text_of[d.card_id]) for d in ok}
    entities, key2id = resolve_entities({c: m for c, (m, _f) in results.items()},
                                        adapter.duplicate_groups)
    mentions, facts = assemble(results, key2id, validity_windows(rows))
    onto = adapter.ontology(entities, facts)
    classes = [(c.get("name") or c.get("label"), c.get("subClassOf") or c.get("parent"))
               for c in onto.get("classes", []) if c.get("name") or c.get("label")]
    viz = pe.STATE_DIR / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    (viz / f"ontology-{safe(source)}.json").write_text(json.dumps(onto, default=str, indent=1),
                                                       encoding="utf-8")
    write_back(store, source, [d.card_id for d in ok], entities, mentions, facts, classes)
    typed = sum(1 for f in facts if f[2] in pe.ONTO_RELATIONS)
    pe.log(f"  wrote {len(ok)} cards: {len(entities)} entities, {len(mentions)} mentions, "
           f"{len(facts)} facts ({typed} typed), {len(classes)} ontology classes "
           f"in {time.time() - t0:.0f}s  [source={source}]")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ingest_semantica.py",
                                 description="Ontology ingestion with Semantica.")
    ap.add_argument("--method", choices=["regex", "llm"], default="regex")
    ap.add_argument("--llm", default="anthropic:claude-haiku-4-5",
                    help="provider:model for --method llm (key read from .env)")
    ap.add_argument("--extract-limit", type=int, default=None,
                    help="cards to read (default: all for regex, 300 for llm; 0 = all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    ap.add_argument("--database-url", default=None)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    store = pe.Store(pe.Config(database_url=a.database_url))
    rows = fetch_cards(store)
    if not rows:
        raise SystemExit("no cards in the database; run `uv run proofline_eval.py seed` first")
    run_extraction(store, rows, a, source_id(a.method, a.llm if a.method == "llm" else None))
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
