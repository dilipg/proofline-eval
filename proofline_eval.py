#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.26",
#   "psycopg[binary]>=3.1",
#   "pgserver>=0.1.4; sys_platform == 'darwin' or platform_machine == 'x86_64' or platform_machine == 'AMD64'",
#   "httpx>=0.27",
#   "langfuse>=4.0",
#   "sentence-transformers>=3.0",
#   "adapters>=1.0",
# ]
# ///
"""
proofline_eval.py: a single-file retrieval evaluation harness.

Problem it answers
------------------
Two candidate scorers rank pieces of a shared research record for a query.
Decide which one is better, on a record nobody has labelled.

It implements all four branches of that problem as runnable experiments:

  B1  Gold from nowhere   Where do labels come from, and how wrong is each source?
  B2  Judge or metric     Ranking metrics vs an LLM judge, and the judge's own biases.
  B3  The long record     Does the scorer survive scale and hard distractors?
  B4  Run it every day    The loop: paired deltas, slices, power, and a release gate.

Design positions this file takes (argue with them, they are the interesting part)
---------------------------------------------------------------------------------
 1. Selection, not measurement. Everything is a PAIRED comparison of two scorers on the
    same queries. We never report an absolute quality number as if it meant something.
 2. The labels already exist as links. `qrels_dag_mined` derives graded relevance from
    the record's own parent/citation edges, which are relevance judgments the authors
    already made. B1 measures how badly that is biased, instead of assuming.
 3. Incomplete judgments are not negative judgments. Anything mined is incomplete, so
    the default metric is condensed-list nDCG and bpref, never raw recall@k.
 4. Supersession is a filter problem, not a ranking problem. Near-identical text cannot
    be separated by a semantic scorer; it is separated by metadata. So it is a HARD
    CHECK that blocks on one occurrence, not a metric to trade off.
 5. A frozen test set goes wrong, not just stale. Every query carries an `as_of`, and
    retrieval is evaluated against the corpus snapshot at that time.
 6. Circularity is checked, not remembered. A scorer that reads graph structure is
    refused against graph-derived labels.
 7. Underpowered is a verdict. Every slice reports its CI and its minimum detectable
    effect; a slice that cannot resolve the effect is labelled, not silently trusted.

Quickstart
----------
    uv run proofline_eval.py all --profile quick

    # with real models (put keys in .env, never on the command line)
    uv run proofline_eval.py all --profile quick \
        --embedder sentence-transformers --judge openai,anthropic

Storage is Postgres. With no DATABASE_URL it boots a private embedded server via
pgserver (no Docker, no install). Set DATABASE_URL to use your own.

Tracing is Langfuse (optional). Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from itertools import cycle
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------------
# §0  Config
# ----------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("PROOFLINE_STATE", HERE / ".proofline")).resolve()


def load_dotenv(path: Path) -> None:
    """Minimal .env reader. No dependency, no surprises, does not overwrite real env."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


load_dotenv(HERE / ".env")
load_dotenv(Path.cwd() / ".env")


PROFILES = {
    # name        cards  prooflines  queries/source  scale points
    "smoke": dict(cards=400, prooflines=18, n_queries=60, scale=[400], mh_queries=20),
    "quick": dict(cards=2500, prooflines=90, n_queries=220, scale=[2500], mh_queries=150),
    "full": dict(cards=12000, prooflines=420, n_queries=600, scale=[2500, 6000, 12000],
                 mh_queries=400),
    # Real-record scale run. Pushes brute-force cosine (design position 7) to 60k x 768
    # -- ~184MB scanned per query -- which tests that position instead of asserting it.
    "arxiv-scale": dict(cards=60000, prooflines=0, n_queries=2000,
                        scale=[6000, 20000, 60000], mh_queries=0),
}


@dataclass
class Config:
    profile: str = "quick"
    source: str = "synthetic"              # synthetic | arxiv
    eval_provenance: str = "planted"       # which query set B2/B3/B4 evaluate on
    seed: int = 20260909
    embedder: str = "hash"                 # hash | sentence-transformers | openai
    embed_dim: int = 256
    judges: tuple[str, ...] = ("mock",)    # mock | openai | anthropic
    judge_pairs: int = 60                  # how many query-pairs to send to each judge
    topk: int = 10
    bootstrap: int = 2000
    target_effect: float = 0.02            # smallest nDCG delta worth shipping
    slice_tolerance: float = -0.02         # a slice CI entirely below this blocks
    abstain: bool = True                   # apply a calibrated no-answer threshold
    max_eval_queries: int = 1500            # cap per provenance, for tractability
    scale_judged_budget: int = 2000         # see branch3_scale: pinned-document budget
    abstain_fabr: float = 0.10             # matched false-abstention rate for calibration
    entities: bool = True                  # plant the typed entity layer (synthetic only)
    baseline: str = "bm25"
    candidate: str = "hybrid_rrf"
    database_url: Optional[str] = None
    langfuse: bool = True
    out_dir: Path = field(default_factory=lambda: STATE_DIR / "reports")

    @property
    def p(self) -> dict:
        return PROFILES[self.profile]


def log(msg: str, *, indent: int = 0) -> None:
    print(("  " * indent) + msg, flush=True)


def rule(title: str = "", ch: str = "─", width: int = 92) -> None:
    if title:
        pad = width - len(title) - 3
        print(f"\n{ch * 2} {title} {ch * max(pad, 0)}", flush=True)
    else:
        print(ch * width, flush=True)


# ----------------------------------------------------------------------------------
# §1  Synthetic corpus: a Proofline-shaped record with planted ground truth
# ----------------------------------------------------------------------------------
#
# Why synthetic rather than a Kaggle dump: because we need to know the answer.
# With a real corpus you can measure "scorer A beat scorer B on these labels". You
# cannot measure "and the labels were right". Here we plant the truth, then derive
# labels from the record's own link structure the way a real deployment would have to,
# and measure how far the derived labels drift from the truth. That drift IS branch 1.
#
# What is planted:
#   true_support[card]   the cards that genuinely contain what this card rests on
#   parent_ids[card]     what the author actually wrote down: an incomplete, rank-biased
#                        sample of true_support, plus occasional false links
#   supersedes           real version chains, near-identical text, one current head
#   cross-topic twins    same surface vocabulary, different meaning, different proofline
#   no-answer queries    information needs whose evidence was deliberately withheld
# ----------------------------------------------------------------------------------

CARD_TYPES = ["seed", "finding", "hypothesis", "literature_note", "synthesis"]
STAGES = ["ideation", "lit_review", "hypothesis", "results"]

_SYLL_A = ["ret", "cal", "mod", "gra", "sem", "lat", "quan", "phon", "meta", "iso",
           "trans", "endo", "hyper", "sub", "poly", "mono", "cyto", "neuro", "geo", "thermo"]
_SYLL_B = ["riev", "ibra", "ula", "dien", "anti", "enc", "til", "emic", "stab", "morph",
           "duct", "plas", "genic", "valu", "trop", "lysis", "kinet", "phase", "vector", "field"]
_SYLL_C = ["al", "ic", "ion", "ance", "ity", "ing", "ment", "ure", "ency", "osis"]


def make_vocab(rng: random.Random, n: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        w = rng.choice(_SYLL_A) + rng.choice(_SYLL_B) + rng.choice(_SYLL_C)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


CONNECTIVE = (
    "we observe that the {a} of {b} does not hold once {c} is controlled for , which "
    "suggests {d} rather than {e} . the effect is stable across {f} but collapses under "
    "{g} . measuring {h} directly would settle whether {i} is doing the work ."
).split()


@dataclass
class Card:
    id: str
    proofline_id: str
    topic: int
    kc_type: str
    kc_stage: str
    title: str
    one_line: str
    body: str
    committed_at: datetime
    version: int
    supersedes_id: Optional[str]
    is_current: bool
    parent_ids: list[str]           # observable: what the author recorded
    citation_ids: list[str]         # observable: cited pieces
    true_support: list[str]         # PLANTED: what actually grounds this card
    terms: list[str]
    inherited_terms: list[str]      # PLANTED: the words that came from its support
    findability: float              # PLANTED: how easy this card is to stumble on
    true_mentions: list = field(default_factory=list)  # PLANTED: (entity_id, surface, role)
    true_facts: list = field(default_factory=list)     # PLANTED: (subj, rel, obj, value)

    @property
    def text(self) -> str:
        return f"{self.title} . {self.one_line} . {self.body}"


@dataclass
class Query:
    id: str
    text: str
    provenance: str                 # planted | dag_mined | generated | usage_mined
    as_of: datetime
    source_card: Optional[str]
    rel: dict[str, float]           # card_id -> graded relevance (0 excluded)
    slices: list[str]
    answerable: bool
    intent: str = "current"         # current | history (declared, like as_of)
    family: Optional[str] = None    # multihop family, or None for the older provenances
    slots: list = field(default_factory=list)      # [set of acceptable card ids] per hop
    answer: Optional[dict] = None   # gold answer (P3 scores it); multihop only
    pair_id: Optional[str] = None   # contrast group: twins, chrono halves, aggregate pairs


@dataclass
class Corpus:
    cards: list[Card]
    by_id: dict[str, Card]
    queries: dict[str, list[Query]]           # provenance -> queries
    held_out_prooflines: set[str]
    topics: dict[int, list[str]]
    stats: dict[str, Any]
    entities: list = field(default_factory=list)       # PLANTED: (id, name, kind, parent_kind)


def _paraphrase(terms: Sequence[str], rng: random.Random) -> list[str]:
    """Return terms mangled so lexical overlap drops but meaning (topic) is preserved.

    This is what makes the `planted` query set hard: a real user asks in their own
    words. A generated query set skips this step, which is exactly why it is easy.
    """
    out = []
    for t in terms:
        r = rng.random()
        if r < 0.62:
            out.append(t)                        # kept verbatim
        elif r < 0.86:
            out.append(t[: max(5, len(t) - 2)])  # stemmed / truncated
        else:
            out.append(t[::-1][:6])              # a synonym: same concept, no overlap
    return out


# ---- planted entity layer ---------------------------------------------------------
#
# Typed entities and the relations between them, rendered into card text so an
# extractor has something real to find, and recorded as truth so B1 can grade whatever
# it found. The ontology DESIGN (classes, relations, roles) is shared with
# ingest_semantica.py. The INSTANCES are truth, and live only in Card.true_* and the
# planted tables.
#
# It runs after the queries are built and draws from its own random stream, so
# --no-entities reproduces the corpus byte for byte. No edge is ever added: a "linked"
# placement reuses a citation that already exists.
# ----------------------------------------------------------------------------------

ONTO_CLASSES: dict[str, Optional[str]] = {
    "Method": None, "VariationalMethod": "Method", "KernelMethod": "Method",
    "SamplingMethod": "Method", "Dataset": None, "Benchmark": "Dataset",
    "Corpus": "Dataset", "Metric": None, "Material": None, "Organism": None,
}
# entity -> entity relations; domain and range are TOP-LEVEL classes
ONTO_RELATIONS: dict[str, tuple[frozenset, frozenset]] = {
    "extends": (frozenset({"Method"}), frozenset({"Method"})),
    "evaluated_on": (frozenset({"Method"}), frozenset({"Dataset"})),
    "measured_by": (frozenset({"Dataset"}), frozenset({"Metric"})),
    "reports": (frozenset({"Method"}), frozenset({"Dataset"})),
}
# card -> entity roles, strongest first
MENTION_ROLES = ("introduces", "uses", "studies", "mentions")


def top_class(kind: str) -> str:
    while ONTO_CLASSES.get(kind):
        kind = ONTO_CLASSES[kind]
    return kind


_ENT_A = ["kav", "orv", "dor", "pel", "tril", "zan", "quor", "vesp", "hal", "mir",
          "tob", "xen", "lum", "brak", "cyr", "fen", "gol", "nim", "sav", "ulm"]
_ENT_B = ["rel", "ane", "vin", "lia", "on", "dex", "ith", "ar", "ova", "us",
          "en", "ix", "ost", "ula", "ep"]
_KIND_WORD = {"VariationalMethod": "variational", "KernelMethod": "kernel",
              "SamplingMethod": "sampling", "Benchmark": "benchmark", "Corpus": "corpus"}
_DATASET_SUFFIX = {"Benchmark": ("QA", "Bench"), "Corpus": ("Corpus", "Text")}
_MATERIAL_SUFFIX = ("oxide", "nitride", "alloy", "polymer")
# At least three phrasings per relation, so no extractor wins on a single template.
_SAY: dict[str, tuple[str, ...]] = {
    "intro_method": ("we introduce {e} , a {k} method", "{e} is a {k} method proposed in this note",
                     "this note presents {e} , a {k} approach"),
    "intro_dataset": ("we release {e} , a new {k}", "{e} is a {k} assembled for this study",
                      "the {k} {e} is introduced here"),
    "intro_material": ("we synthesise {e} for the first time", "{e} is a new material prepared here",
                       "this work reports the preparation of {e}"),
    "measured_by": ("{d} is scored with {x}", "performance on {d} is measured in {x}",
                    "{d} reports results as {x}"),
    "uses": ("we adopt {e} here", "our pipeline builds on {e}", "the analysis relies on {e}"),
    "studies": ("experiments use {e} as the model organism", "we study {e}",
                "samples of {e} were examined"),
    "reports": ("{e} reaches {v} {x} on {d}", "on {d} , {e} scores {v} {x}",
                "we measure {v} {x} for {e} on {d}"),
    "extends": ("{e} extends {f}", "{e} builds directly on {f}", "{e} is a refinement of {f}"),
    "evaluated_on": ("{e} is evaluated on {d}", "we test {e} on {d}",
                     "{d} serves as the testbed for {e}"),
}


# P2's history and chrono_asof families need results that change across a card's own
# revisions. Left to chance, a reporter lands on a version chain ~4% of the time, so most
# methods get one reporter placed on a chain, and that reporter's value changes more often.
CHAIN_REPORT_P = 0.9
CHAIN_CHANGE_P = 0.8
# Methods reported on several datasets feed aggregate_set; single-dataset methods feed
# every other family (their answers must have one referent). P2 tunes this share.
MULTI_DATASET_P = 0.35
# Methods that extend an earlier one feed bridge3 (one more hop, through `extends`).
EXTENDS_P = 0.45


def plant_entities(cards: list[Card], vocab: set[str], seed: int
                   ) -> tuple[list[tuple[str, str, str, Optional[str]]], dict]:
    rng = random.Random(seed + 1)
    by_id = {c.id: c for c in cards}
    succ = {c.supersedes_id: c.id for c in cards if c.supersedes_id}

    def versions(root: Card) -> list[Card]:
        out, cid = [root], root.id
        while cid in succ:
            cid = succ[cid]
            out.append(by_id[cid])
        return out

    roots = sorted((c for c in cards if c.supersedes_id is None),
                   key=lambda c: (c.committed_at, c.id))
    root_ts = [c.committed_at for c in roots]
    nonseed = [c for c in roots if c.kc_type != "seed"] or roots

    def later_than(c: Card) -> list[Card]:
        return roots[bisect.bisect_right(root_ts, c.committed_at):]

    chain_roots = [c for c in roots if c.id in succ]
    chain_ts = [c.committed_at for c in chain_roots]

    def chains_later_than(c: Card) -> list[Card]:
        return chain_roots[bisect.bisect_right(chain_ts, c.committed_at):]

    # time-valid observable links only: a "linked" placement must survive seeding
    cites = {c.id: {d for d in c.parent_ids + c.citation_ids
                    if by_id[d].committed_at <= c.committed_at} for c in cards}
    nbrs: dict[str, set[str]] = defaultdict(set)
    for s, ds in cites.items():
        for d in ds:
            nbrs[s].add(d)
            nbrs[d].add(s)
    # roots that some later root cites: the only introducers a linked bridge can use
    root_ids = {r.id for r in roots}
    cited = {d for r in roots for d in cites[r.id]
             if d in root_ids and by_id[d].committed_at < r.committed_at}

    stems = [a + b for a in _ENT_A for b in _ENT_B]
    rng.shuffle(stems)
    assert not set(stems) & vocab, "entity stems collide with topic vocabulary"
    pool = dict(method=stems[0:60], benchmark=stems[60:135], corpus=stems[135:210],
                material=stems[210:250], genus=stems[250:270], species=stems[270:290],
                metric=stems[290:294])
    n = len(cards)
    counts = dict(method=max(12, n // 10), dataset=max(6, n // 40),
                  material=max(4, n // 80), organism=max(4, n // 80))

    ents: list[tuple[str, str, str, Optional[str]]] = []
    name_of: dict[str, str] = {}
    said: dict[str, list[str]] = defaultdict(list)
    ment: dict[str, dict[str, tuple[set, str]]] = defaultdict(dict)
    facts: dict[str, list[tuple]] = defaultdict(list)
    stats = Counter()

    def new_entity(prefix: str, name: str, kind: str) -> str:
        eid = f"ent_{prefix}_{len(ents):05d}"
        ents.append((eid, name, kind, ONTO_CLASSES[kind]))
        name_of[eid] = name
        return eid

    def say(card: Card, template: str, mentions: list[tuple[str, str, str]],
            fact: Optional[tuple] = None, **slots: str) -> None:
        said[card.id] += template.format(**slots).split() + ["."]
        for eid, surface, role in mentions:
            surfs, prev = ment[card.id].get(eid, (set(), "mentions"))
            surfs.add(surface)
            ment[card.id][eid] = (surfs, min(prev, role, key=MENTION_ROLES.index))
        if fact:
            facts[card.id].append(fact)

    def new_value(old: Optional[str] = None) -> str:
        while True:
            v = f"0.{rng.randrange(40, 99)}"
            if v != old:
                return v

    def uniq(cs: list[Card]) -> list[Card]:
        # Card is an unhashable dataclass: dedupe by id, keeping first-seen order
        return list({c.id: c for c in cs}.values())

    def dataset_names(kind: str):
        # a function, not a comprehension: a generator built in a dict comprehension
        # would read `kind` late and give every class the last class's suffixes
        r = 0
        while True:
            for s in pool[kind.lower()]:
                for suf in _DATASET_SUFFIX[kind]:
                    yield f"{s.title()}-{suf}{r + 1 if r else ''}"
            r += 1

    # metrics, then datasets (each introduced once, scored with one metric)
    metric_ids = [new_entity("x", f"{s}-score", "Metric") for s in pool["metric"]]
    names = {k: dataset_names(k) for k in _DATASET_SUFFIX}
    dataset_ids, metric_of = [], {}
    for i in range(counts["dataset"]):
        kind = ("Benchmark", "Corpus")[i % 2]
        did = new_entity("d", next(names[kind]), kind)
        dataset_ids.append(did)
        metric_of[did] = rng.choice(metric_ids)
        intro = rng.choice(nonseed)
        t_i, t_m = rng.choice(_SAY["intro_dataset"]), rng.choice(_SAY["measured_by"])
        x = metric_of[did]
        for v in versions(intro):
            say(v, t_i, [(did, name_of[did], "introduces")], e=name_of[did], k=_KIND_WORD[kind])
            say(v, t_m, [(did, name_of[did], "mentions"), (x, name_of[x], "mentions")],
                (did, "measured_by", x, None), d=name_of[did], x=name_of[x])

    # methods: introduced once, used later (the bridges), reported on over time
    fresh = cycle(pool["method"])
    stem_of, intro_of, used_names = {}, {}, set()
    for i in range(counts["method"]):
        sibling = rng.choice(list(stem_of)) if stem_of and rng.random() < 0.2 else None
        stem = stem_of[sibling] if sibling else next(fresh)
        while True:
            num = rng.randrange(2, 100)
            name = f"{stem.title()}-{num}"
            if name not in used_names:
                used_names.add(name)
                break
        kind = rng.choice(("VariationalMethod", "KernelMethod", "SamplingMethod"))
        mid = new_entity("m", name, kind)
        stem_of[mid] = stem
        avoid = intro_of[sibling].topic if sibling else None
        # decide linked vs unlinked BEFORE choosing the introducer: a linked bridge needs
        # an introducer somebody later cites, and most cards are cited by nobody
        want_linked = rng.random() < 0.5
        cands = [r for r in nonseed if avoid is None or r.topic != avoid] or nonseed
        linkable = [r for r in cands if r.id in cited] if want_linked else []
        intro = rng.choice(linkable or cands)
        intro_of[mid] = intro
        if sibling or sum(1 for s in stem_of.values() if s == stem) > 1:
            stats["near_miss_methods"] += 1
        # an alias belongs to one method only: 'KA-7' must not name two things
        aliases = ([s for s in (f"{stem.title()} {num}", f"{stem[:2].upper()}-{num}")
                    if s not in used_names] if rng.random() < 0.3 else [])
        used_names.update(aliases)
        stats["aliased_methods"] += bool(aliases)
        tpl = rng.choice(_SAY["intro_method"])
        for v in versions(intro):
            say(v, tpl, [(mid, name, "introduces")], e=name, k=_KIND_WORD[kind])

        older = [m for m in intro_of if m != mid and intro_of[m].committed_at < intro.committed_at]
        if older and rng.random() < EXTENDS_P:
            m2 = rng.choice(older)
            tpl = rng.choice(_SAY["extends"])
            for v in versions(intro):
                say(v, tpl, [(mid, name, "mentions"), (m2, name_of[m2], "mentions")],
                    (mid, "extends", m2, None), e=name, f=name_of[m2])

        multi = rng.random() < MULTI_DATASET_P and len(dataset_ids) >= 2
        ds = (rng.sample(dataset_ids, k=rng.randrange(2, min(8, len(dataset_ids)) + 1))
              if multi else [rng.choice(dataset_ids)])
        if multi:
            for d in ds:
                tpl = rng.choice(_SAY["evaluated_on"])
                for v in versions(intro):
                    say(v, tpl, [(mid, name, "mentions"), (d, name_of[d], "mentions")],
                        (mid, "evaluated_on", d, None), e=name, d=name_of[d])

        later = later_than(intro)
        if not later:
            continue
        a_card = None
        if want_linked:
            linked = [r for r in later if intro.id in cites[r.id]]
            a_card = rng.choice(linked) if linked else None
        if a_card is None:
            far = nbrs[intro.id]
            unlinked = [r for r in later if r.proofline_id != intro.proofline_id
                        and intro.id not in nbrs[r.id] and not (nbrs[r.id] & far)]
            a_card = rng.choice(unlinked or later)
        stats["bridges_linked" if intro.id in cites[a_card.id] else "bridges_unlinked"] += 1
        users = uniq([a_card] + rng.sample(later, k=min(len(later), rng.randrange(0, 4))))
        for u in users:
            surface = rng.choice(aliases) if aliases and rng.random() < 0.5 else name
            tpl = rng.choice(_SAY["uses"])
            for v in versions(u):
                say(v, tpl, [(mid, surface, "uses")], e=surface)

        k_more = rng.randrange(0, 4) + (len(ds) if multi else 0)
        reporters = ([intro] if rng.random() < 0.6 else []) + rng.sample(later, k=min(len(later), k_more))
        taken = {r.id for r in reporters}
        on_chain = [r for r in chains_later_than(intro) if r.id not in taken]
        chain_rep = rng.choice(on_chain) if on_chain and rng.random() < CHAIN_REPORT_P else None
        if chain_rep is not None:
            reporters.append(chain_rep)
            stats["chain_reporters"] += 1
        for j, r in enumerate(uniq(reporters)):
            d = ds[j % len(ds)]
            x = metric_of[d]
            surface = name if r is intro or not aliases or rng.random() >= 0.3 else rng.choice(aliases)
            tpl = rng.choice(_SAY["reports"])
            value = new_value()
            change_p = CHAIN_CHANGE_P if r is chain_rep else 0.5
            for k, v in enumerate(versions(r)):
                if k and rng.random() < change_p:
                    value = new_value(value)
                    stats["value_changes"] += 1
                say(v, tpl, [(mid, surface, "mentions"), (d, name_of[d], "mentions"),
                             (x, name_of[x], "mentions")],
                    (mid, "reports", d, value), e=surface, v=value, x=name_of[x], d=name_of[d])
                facts[v.id].append((mid, "evaluated_on", d, None))

    for i in range(counts["material"]):
        name = f"{pool['material'][i % 40]} {_MATERIAL_SUFFIX[(i // 40) % 4]}"
        mat = new_entity("mat", name, "Material")
        intro = rng.choice(nonseed)
        tpl = rng.choice(_SAY["intro_material"])
        for v in versions(intro):
            say(v, tpl, [(mat, name, "introduces")], e=name)
        later = later_than(intro)
        for u in rng.sample(later, k=min(len(later), rng.randrange(1, 4))):
            tpl = rng.choice(_SAY["uses"])
            for v in versions(u):
                say(v, tpl, [(mat, name, "uses")], e=name)

    for i in range(counts["organism"]):
        name = f"{pool['genus'][i % 20].title()} {pool['species'][(i // 20) % 20]}ii"
        org = new_entity("o", name, "Organism")
        for c in rng.sample(roots, k=min(len(roots), rng.randrange(2, 6))):
            tpl = rng.choice(_SAY["studies"])
            for v in versions(c):
                say(v, tpl, [(org, name, "studies")], e=name)

    n_tmpl = len(CONNECTIVE)
    for cid, words in said.items():
        c = by_id[cid]
        body = c.body.split()
        c.body = " ".join(body[:n_tmpl] + words + body[n_tmpl:])
        c.true_mentions = sorted((eid, s, role) for eid, (surfs, role) in ment[cid].items()
                                 for s in surfs)
        c.true_facts = sorted(set(facts[cid]), key=repr)
    out = dict(stats)
    out["entities"] = dict(Counter(k for _i, _n, k, _p in ents))
    out["true_mentions"] = sum(len(c.true_mentions) for c in cards)
    out["true_facts"] = sum(len(c.true_facts) for c in cards)
    return ents, out


# ---- multi-hop queries (P2) -------------------------------------------------------
#
# Composed from the planted entity layer, MuSiQue-style. A question speaks in card A's
# own vocabulary (paraphrased) plus a relation cue, and never names the bridge entity,
# so the answer's card is reachable only through the entity A uses. Gold is a set of
# card VERSIONS valid at the question's as_of, one set per hop, and the answer itself
# is stored so P3 can score answers without re-deriving them. Its own random stream
# (seed + 2), after everything else: no other query set moves.
# ----------------------------------------------------------------------------------

MH_FAMILIES = ("bridge", "bridge3", "chrono_asof", "timeline", "history",
               "aggregate_set", "aggregate_count")
LAST_HOP_FAMILIES = ("bridge", "bridge3", "chrono_asof")
_MH_CUE = {
    "bridge": "what score did the method used here reach on its {kind}",
    "bridge3": "what score did the method that the approach used here extends reach on its {kind}",
    "chrono_asof": "what score did the method used here reach on its {kind}",
    "timeline": "how have reported results for the method used here changed over time",
    "history": "what did the reported result for the method used here read before it was revised",
    "aggregate_set": "which datasets have results been reported on for the method used here",
    "aggregate_count": "how many results have been reported for the method used here",
}


def build_multihop_queries(cards: list[Card], entities: list[tuple], target: int, seed: int,
                           topk: int) -> tuple[list[Query], dict]:
    rng = random.Random(seed + 2)
    by_id = {c.id: c for c in cards}
    succ = {c.supersedes_id: c for c in cards if c.supersedes_id}
    end = max(c.committed_at for c in cards)
    kind = {e[0]: e[2] for e in entities}
    ename = {e[0]: e[1] for e in entities}
    minute = timedelta(minutes=1)

    def root_of(c: Card) -> Card:
        while c.supersedes_id:
            c = by_id[c.supersedes_id]
        return c

    def chain(root: Card) -> list[Card]:
        out = [root]
        while out[-1].id in succ:
            out.append(succ[out[-1].id])
        return out

    def valid(c: Card, t: datetime) -> bool:
        nxt = succ.get(c.id)
        return c.committed_at <= t and (nxt is None or nxt.committed_at > t)

    def version_at(root: Card, t: datetime) -> Optional[Card]:
        return next((v for v in chain(root) if valid(v, t)), None)

    reports: dict[str, list[tuple[Card, str, str]]] = defaultdict(list)
    users: dict[str, dict[str, Card]] = defaultdict(dict)
    extends: dict[str, tuple[Card, str]] = {}
    for c in cards:
        for s, r, o, v in c.true_facts:
            if r == "reports":
                reports[s].append((c, o, v))
            elif r == "extends" and c.supersedes_id is None:
                extends.setdefault(s, (c, o))
        if c.supersedes_id is None:
            for e, _surf, role in c.true_mentions:
                if role == "uses" and top_class(kind[e]) == "Method":
                    users[e][c.id] = c

    def reports_at(m: str, t: datetime) -> dict[str, tuple[Card, str, str]]:
        """Report facts of `m` valid at t, one per reporting card chain (keyed by root)."""
        return {root_of(c).id: (c, d, v) for c, d, v in reports[m] if valid(c, t)}

    def first_report(m: str) -> Optional[datetime]:
        return min((c.committed_at for c, _d, _v in reports[m]), default=None)

    def latest(m: str, t: datetime) -> Optional[tuple[Card, str, str]]:
        r = reports_at(m, t)
        if not r or len({d for _c, d, _v in r.values()}) != 1:
            return None
        return max(r.values(), key=lambda x: (x[0].committed_at, x[0].id))

    def stale(m: str, d: str, t: datetime) -> list[str]:
        return sorted({v for c, dd, v in reports[m]
                       if dd == d and c.id in succ and succ[c.id].committed_at <= t})

    def question(a: Card, cue: str) -> str:
        own = [w for w in a.terms if w not in a.inherited_terms] or a.terms
        pick = rng.sample(own, k=min(len(own), rng.randrange(4, 8)))
        return " ".join(_paraphrase(pick, rng)) + " " + cue

    def linked(a: Card, gold: Card) -> bool:
        return gold.id in (a.parent_ids + a.citation_ids) and gold.committed_at <= a.committed_at

    def alias_used(gold: Card, m: str) -> bool:
        return any(e == m and s != ename[m] for e, s, _r in gold.true_mentions)

    out: list[Query] = []
    stats: Counter = Counter()
    cap = 2 * target

    def emit(family: str, a: Card, t: datetime, slots: list[set], answer: dict,
             text: str, intent: str = "current", pair_id: Optional[str] = None,
             gold: Optional[Card] = None) -> Optional[Query]:
        flat = [cid for s in slots for cid in s]
        if len(flat) != len(set(flat)):
            stats["skipped_shared_card"] += 1
            return None
        slices = [family, "hops3" if family == "bridge3" else "hops2"]
        if gold is not None and family in LAST_HOP_FAMILIES:
            slices.append("linked" if linked(a, gold) else "unlinked")
            if alias_used(gold, answer["_m"]):
                slices.append("alias")
            if gold.proofline_id != a.proofline_id:
                slices.append("cross_project")
        if family.startswith("aggregate") and len(slots) - 1 > topk:
            slices.append("set_gt_k")
        answer = {k: v for k, v in answer.items() if not k.startswith("_")}
        q = Query(id=f"q_mh_{family}_{stats[family]:05d}", text=text, provenance="multihop",
                  as_of=t, source_card=a.id, rel={cid: 2.0 for cid in flat}, slices=slices,
                  answerable=True, intent=intent, family=family, slots=slots, answer=answer,
                  pair_id=pair_id)
        stats[family] += 1
        out.append(q)
        return q

    def twin(q: Query, a_root: Card, m: str) -> None:
        f = first_report(m)
        if f is None or a_root.committed_at >= f:
            return
        t = a_root.committed_at + (f - a_root.committed_at) / 2
        a = version_at(a_root, t)
        if a is None:
            return
        q.pair_id = q.id
        out.append(Query(id=f"{q.id}_twin", text=q.text, provenance="multihop", as_of=t,
                         source_card=a.id, rel={}, slices=["no_answer", "twin"],
                         answerable=False, family="twin", pair_id=q.id))
        stats["twin"] += 1

    pairs = [(m, u) for m in sorted(users) if reports[m] for u in sorted(users[m])]
    rng.shuffle(pairs)

    for m, uid in pairs:
        a_root = users[m][uid]
        lower = max(a_root.committed_at, first_report(m))

        if stats["bridge"] < cap:
            t = min(end, lower + timedelta(days=rng.randrange(0, 60)))
            a, got = version_at(a_root, t), latest(m, t)
            if a and got and root_of(got[0]).id != a_root.id:
                g, d, v = got
                q = emit("bridge", a, t, [{a.id}, {g.id}],
                         {"value": v, "stale": stale(m, d, t), "_m": m},
                         question(a, _MH_CUE["bridge"].format(kind=_KIND_WORD[kind[d]])), gold=g)
                if q:
                    twin(q, a_root, m)

        if stats["chrono_asof"] < cap:
            events = sorted({c.committed_at for c, _d, _v in reports[m]} |
                            {succ[c.id].committed_at for c, _d, _v in reports[m] if c.id in succ})
            for e in events:
                t1, t2 = e - minute, e + minute
                if t1 < lower or t2 > end:
                    continue
                r1, r2 = latest(m, t1), latest(m, t2)
                a1, a2 = version_at(a_root, t1), version_at(a_root, t2)
                if not (r1 and r2 and a1 and a2) or r1[2] == r2[2] or r1[1] != r2[1]:
                    continue
                if root_of(r1[0]).id == a_root.id or root_of(r2[0]).id == a_root.id:
                    continue
                text = question(a2, _MH_CUE["chrono_asof"].format(kind=_KIND_WORD[kind[r1[1]]]))
                pid = f"pair_chrono_{stats['chrono_asof']:05d}"
                emit("chrono_asof", a1, t1, [{a1.id}, {r1[0].id}],
                     {"value": r1[2], "stale": stale(m, r1[1], t1), "_m": m}, text,
                     pair_id=pid, gold=r1[0])
                emit("chrono_asof", a2, t2, [{a2.id}, {r2[0].id}],
                     {"value": r2[2], "stale": stale(m, r2[1], t2), "_m": m}, text,
                     pair_id=pid, gold=r2[0])
                break

        if stats["timeline"] < cap:
            a, r = version_at(a_root, end), reports_at(m, end)
            if a and len(r) >= 2 and len({d for _c, d, _v in r.values()}) == 1:
                rs = sorted(r.values(), key=lambda x: (x[0].committed_at, x[0].id))
                emit("timeline", a, end, [{a.id}] + [{c.id} for c, _d, _v in rs],
                     {"values": [v for _c, _d, v in rs]},
                     question(a, _MH_CUE["timeline"]))

        if stats["history"] < cap:
            changed = [root_of(c) for c, _d, _v in reports[m]]
            roots = {r.id: r for r in changed if len({v for c, _d, v in reports[m]
                                                      if root_of(c).id == r.id}) > 1}
            ds = {d for _c, d, _v in reports[m]}
            if len(roots) == 1 and len(ds) == 1:
                (rroot,) = roots.values()
                last = chain(rroot)[-1]
                t = min(end, max(last.committed_at, a_root.committed_at) + timedelta(days=rng.randrange(0, 30)))
                a = version_at(a_root, t)
                vs = [v for v in chain(rroot) if v.committed_at <= t]
                vals = [next(val for s, r_, _o, val in v.true_facts if s == m and r_ == "reports")
                        for v in vs]
                if a and rroot.id != a_root.id and len(set(vals)) > 1:
                    emit("history", a, t, [{a.id}] + [{v.id} for v in vs], {"values": vals},
                         question(a, _MH_CUE["history"]), intent="history")

        for fam in ("aggregate_set", "aggregate_count"):
            if stats[fam] >= cap:
                continue
            a, r = version_at(a_root, end), reports_at(m, end)
            if not a or not 2 <= len(r) <= topk - 1:
                continue
            by_d: dict[str, set] = defaultdict(set)
            for c, d, _v in r.values():
                by_d[d].add(c.id)
            if fam == "aggregate_set" and not 2 <= len(by_d) <= 8:
                continue
            if fam == "aggregate_set":
                order = sorted(by_d, key=lambda d: min(by_id[c].committed_at for c in by_d[d]))
                slots = [{a.id}] + [by_d[d] for d in order]
                answer = {"set": sorted(ename[d] for d in order)}
            else:
                slots = [{a.id}] + [{c.id} for c, _d, _v in
                                    sorted(r.values(), key=lambda x: x[0].committed_at)]
                answer = {"count": len(r)}
            emit(fam, a, end, slots, answer, question(a, _MH_CUE[fam]))

    for m1, (c_root, m2) in sorted(extends.items()):
        if stats["bridge3"] >= cap:
            break
        if not reports[m2]:
            continue
        for uid in sorted(users.get(m1, {})):
            if stats["bridge3"] >= cap:
                break
            a_root = users[m1][uid]
            lower = max(a_root.committed_at, c_root.committed_at, first_report(m2))
            t = min(end, lower + timedelta(days=rng.randrange(0, 60)))
            a, c, got = version_at(a_root, t), version_at(c_root, t), latest(m2, t)
            if not (a and c and got):
                continue
            g, d, v = got
            if len({a_root.id, c_root.id, root_of(g).id}) < 3:
                continue
            q = emit("bridge3", a, t, [{a.id}, {c.id}, {g.id}],
                     {"value": v, "stale": stale(m2, d, t), "_m": m2},
                     question(a, _MH_CUE["bridge3"].format(kind=_KIND_WORD[kind[d]])), gold=g)
            if q:
                twin(q, a_root, m2)

    return out, dict(stats)


def build_corpus(cfg: Config, n_cards: int, n_prooflines: int) -> Corpus:
    rng = random.Random(cfg.seed)
    t0 = datetime(2025, 1, 6, tzinfo=timezone.utc)

    n_topics = max(6, n_prooflines // 6)
    vocab = make_vocab(rng, n_topics * 26)
    topics: dict[int, list[str]] = {}
    for t in range(n_topics):
        core = vocab[t * 26:(t + 1) * 26]
        # confusable terms: every topic shares 4 terms with its neighbour, used in a
        # different sense. This is the cross-project distractor from branch 3.
        nb = vocab[((t + 1) % n_topics) * 26:((t + 1) % n_topics) * 26 + 4]
        topics[t] = core + nb

    # one topic is withheld entirely: its queries are the known no-answer set
    withheld_topic = n_topics - 1

    cards: list[Card] = []
    by_id: dict[str, Card] = {}
    per_line = max(6, n_cards // n_prooflines)
    seq = 0

    def new_id() -> str:
        nonlocal seq
        seq += 1
        return f"kc_{seq:07d}"

    proofline_ids: list[str] = []

    for pi in range(n_prooflines):
        topic = rng.randrange(n_topics - 1)     # never the withheld topic
        pl = f"pl_{pi:05d}"
        proofline_ids.append(pl)
        base_t = t0 + timedelta(days=rng.randrange(0, 500), hours=rng.randrange(0, 24))
        pool = topics[topic]
        line_cards: list[Card] = []

        for j in range(per_line):
            if len(cards) >= n_cards:
                break
            kc_type = ("seed" if j == 0 else
                       "synthesis" if j == per_line - 1 else
                       rng.choice(["finding", "hypothesis", "finding", "literature_note"]))
            terms = rng.sample(pool, k=min(len(pool), rng.randrange(5, 9)))

            # ---- planted truth: which existing cards genuinely ground this one
            candidates = line_cards[:]
            true_support: list[str] = []
            inherited: list[str] = []
            if candidates:
                k = min(len(candidates), rng.choice([1, 2, 2, 3]))
                picked = rng.sample(candidates, k)
                true_support = [c.id for c in picked]
                # inherit vocabulary from what it genuinely rests on
                for c in picked:
                    got = rng.sample(c.terms, k=min(3, len(c.terms)))
                    terms += got
                    inherited += got
            # ~12% of cards genuinely rest on something in ANOTHER proofline
            if cards and rng.random() < 0.12:
                other = rng.choice(cards)
                if other.proofline_id != pl:
                    true_support.append(other.id)
                    got = rng.sample(other.terms, k=min(2, len(other.terms)))
                    terms += got
                    inherited += got

            terms = list(dict.fromkeys(terms))
            body_words = []
            for w in CONNECTIVE:
                body_words.append(rng.choice(terms) if w.startswith("{") else w)
            # One latent variable drives two things at once, which is the whole point:
            # a card that dwells on its subject is BOTH easier for a term-matching
            # scorer to surface AND more memorable, so more likely to be the one the
            # author remembered to link. The rank bias on card 6SNVC4 then emerges from
            # the world instead of being asserted by us, and it is measured in B1
            # rather than assumed.
            findability = rng.random()
            body = " ".join(body_words)
            for _ in range(int(6 + 40 * findability)):
                body += " " + rng.choice(terms)

            cid = new_id()
            committed = base_t + timedelta(days=2 * j + rng.randrange(0, 3),
                                           hours=rng.randrange(0, 24))
            card = Card(
                id=cid, proofline_id=pl, topic=topic, kc_type=kc_type,
                kc_stage=rng.choice(STAGES),
                title=" ".join(rng.sample(terms, k=min(3, len(terms)))).title(),
                one_line=" ".join(rng.sample(terms, k=min(5, len(terms)))),
                body=body, committed_at=committed, version=1, supersedes_id=None,
                is_current=True, parent_ids=[], citation_ids=[],
                true_support=true_support, terms=terms,
                inherited_terms=list(dict.fromkeys(inherited)),
                findability=findability,
            )
            cards.append(card)
            by_id[cid] = card
            line_cards.append(card)

        if len(cards) >= n_cards:
            break

    # ---- observable link structure: a noisy, rank-biased sample of the planted truth
    #
    # The author records ~72% of what actually grounded the card, and is MORE likely to
    # record the pieces that are easy to find by lexical search (high overlap with the
    # new card's own words). That is card 6SNVC4's rank bias, planted so we can measure
    # it rather than argue about it. They also record ~8% spurious links.
    for c in cards:
        rec: list[str] = []
        for sid in c.true_support:
            s = by_id[sid]
            p_record = 0.22 + 0.66 * s.findability    # findable -> remembered -> linked
            if rng.random() < min(0.93, p_record):
                rec.append(sid)
        if rng.random() < 0.08 and cards:
            spurious = rng.choice(cards)
            if spurious.id != c.id and spurious.committed_at < c.committed_at:
                rec.append(spurious.id)
        rec = list(dict.fromkeys(rec))
        # A recorded link lands either as a parent (strong) or a citation (weak).
        # Both are observable; what the author never wrote down stays invisible, which
        # is the whole point of the incompleteness this harness has to survive.
        cits = [x for x in rec if rng.random() < 0.22]
        c.citation_ids = cits[:2]
        c.parent_ids = [x for x in rec if x not in c.citation_ids]

    # ---- supersession chains: near-identical text, only the head is current
    n_chains = max(4, len(cards) // 25)
    for _ in range(n_chains):
        head = rng.choice(cards)
        if head.supersedes_id or head.kc_type == "seed":
            continue
        prev_id = head.id
        for v in range(2, rng.choice([2, 2, 3, 4])):
            cid = new_id()
            drift = rng.sample(head.terms, k=min(2, len(head.terms)))
            newer = Card(
                id=cid, proofline_id=head.proofline_id, topic=head.topic,
                kc_type=head.kc_type, kc_stage=head.kc_stage,
                title=head.title, one_line=head.one_line,
                body=head.body + " revised . " + " ".join(drift),
                committed_at=head.committed_at + timedelta(days=7 * (v - 1)),
                version=v, supersedes_id=prev_id, is_current=True,
                parent_ids=list(head.parent_ids), citation_ids=list(head.citation_ids),
                true_support=list(head.true_support), terms=head.terms + drift,
                inherited_terms=list(head.inherited_terms),
                findability=head.findability,
            )
            by_id[prev_id].is_current = False
            cards.append(newer)
            by_id[cid] = newer
            prev_id = cid

    # ---- queries ------------------------------------------------------------------
    n_q = cfg.p["n_queries"]
    current = [c for c in cards if c.is_current and c.true_support]
    rng.shuffle(current)

    held_out = set(rng.sample(proofline_ids, k=max(1, len(proofline_ids) // 5)))

    def grade(qcard: Card, sid: str) -> float:
        s = by_id.get(sid)
        if s is None:
            return 0.0
        return 2.0 if s.proofline_id == qcard.proofline_id else 2.0

    queries: dict[str, list[Query]] = defaultdict(list)

    # (a) PLANTED: the honest set. Query paraphrases the information need; relevance is
    #     the true support, plus one grade of transitive support. This is the yardstick
    #     the other three label sources are judged against. In production it does not
    #     exist; that is the whole problem. Here it does, so we can score the scorers'
    #     scorers.
    for i, c in enumerate(current[:n_q]):
        # A real information need is phrased around the evidence the person is
        # looking for, not around the note they are about to write. So the query is
        # built from the vocabulary this card inherited from what actually grounds it,
        # plus a little of its own framing, then paraphrased.
        want = rng.choice([3, 4, 5, 5, 7, 8, 9, 11])
        base = c.inherited_terms or c.terms
        pick = rng.sample(base, k=min(want, len(base)))
        if len(pick) < want and c.terms:
            pick += rng.sample(c.terms, k=min(want - len(pick), len(c.terms)))
        need = _paraphrase(list(dict.fromkeys(pick)), rng)
        rel: dict[str, float] = {}
        for sid in c.true_support:
            rel[sid] = grade(c, sid)
            for gid in by_id[sid].true_support:     # grandparents: marginal
                rel.setdefault(gid, 1.0)
        rel = {k: v for k, v in rel.items() if by_id[k].committed_at < c.committed_at}
        if not rel:
            continue
        cross = len({by_id[k].proofline_id for k in rel}) > 1
        recent = any((c.committed_at - by_id[k].committed_at).days <= 3 for k in rel)
        qtext = " ".join(need)
        queries["planted"].append(Query(
            id=f"q_pl_{i:05d}", text=qtext, provenance="planted", as_of=c.committed_at,
            source_card=c.id, rel=rel,
            slices=(["short" if len(need) < 6 else "long"]
                    + (["cross_project"] if cross else [])
                    + (["recently_changed"] if recent else [])),
            answerable=True))

    # (b) DAG-MINED: the proposal from the brief. Query = the card's one-liner (short,
    #     abstracted, NOT its body). Relevance = the links the author recorded.
    #     Incomplete and rank-biased by construction; B1 measures by how much.
    for i, c in enumerate(current[:n_q]):
        rel = {pid: 2.0 for pid in c.parent_ids if by_id[pid].committed_at < c.committed_at}
        rel.update({cid_: 1.0 for cid_ in c.citation_ids
                    if by_id[cid_].committed_at < c.committed_at})
        if not rel:
            continue
        toks = c.one_line.split()
        queries["dag_mined"].append(Query(
            id=f"q_dg_{i:05d}", text=c.one_line, provenance="dag_mined",
            as_of=c.committed_at, source_card=c.id, rel=rel,
            slices=["short" if len(toks) < 6 else "long"], answerable=True))

    # (c) GENERATED: a model writes a query from a piece, in the piece's own words.
    #     Known-item, single relevant doc, huge lexical overlap. Smoke test, not a
    #     benchmark: card REVVD5's point, made measurable.
    for i, c in enumerate(current[:n_q]):
        qt = " ".join(rng.sample(c.body.split(), k=min(9, len(c.body.split()))))
        queries["generated"].append(Query(
            id=f"q_gn_{i:05d}", text=qt, provenance="generated", as_of=c.committed_at,
            source_card=c.id, rel={c.id: 2.0},
            slices=["long"], answerable=True))

    # (d) USAGE-MINED: a simulated click log. The "user" only ever saw the incumbent's
    #     top 3, so the label is whatever the incumbent surfaced that was also true.
    #     Rewards the incumbent by construction. Filled in after ingestion (needs the
    #     incumbent's rankings), see build_usage_mined().

    # (e) NO-ANSWER: needs from the withheld topic. Correct behaviour is to return
    #     nothing. Scored as abstention, never as nDCG.
    wpool = topics[withheld_topic]
    latest = max(c.committed_at for c in cards)
    for i in range(max(30, n_q // 4)):
        need = _paraphrase(rng.sample(wpool, k=rng.randrange(4, 8)), rng)
        q = Query(id=f"q_na_{i:05d}", text=" ".join(need), provenance="planted",
                  as_of=latest, source_card=None, rel={},
                  slices=["no_answer", "short" if len(need) < 6 else "long"],
                  answerable=False)
        queries["planted"].append(q)

    entities, ent_stats = (plant_entities(cards, {w for ws in topics.values() for w in ws},
                                          cfg.seed) if cfg.entities else ([], {}))
    if cfg.entities and cfg.p.get("mh_queries"):
        mh, mh_stats = build_multihop_queries(cards, entities, cfg.p["mh_queries"], cfg.seed,
                                              cfg.topk)
        queries["multihop"] = mh
        ent_stats = {**ent_stats, "multihop": mh_stats}
    stats = {
        "cards": len(cards),
        "current_cards": sum(1 for c in cards if c.is_current),
        "superseded": sum(1 for c in cards if not c.is_current),
        "prooflines": len(proofline_ids),
        "held_out_prooflines": len(held_out),
        "mean_true_support": round(
            statistics.fmean([len(c.true_support) for c in cards]), 2),
        "mean_recorded_parents": round(
            statistics.fmean([len(c.parent_ids) for c in cards]), 2),
        "link_recall": round(
            sum(len((set(c.parent_ids) | set(c.citation_ids)) & set(c.true_support))
                for c in cards)
            / max(1, sum(len(c.true_support) for c in cards)), 3),
        "link_precision": round(
            sum(len((set(c.parent_ids) | set(c.citation_ids)) & set(c.true_support))
                for c in cards)
            / max(1, sum(len(set(c.parent_ids) | set(c.citation_ids)) for c in cards)), 3),
    }
    stats.update(ent_stats)
    return Corpus(cards=cards, by_id=by_id, queries=dict(queries),
                  held_out_prooflines=held_out, topics=topics, stats=stats,
                  entities=entities)


# ----------------------------------------------------------------------------------
# §1b  Real corpus: arXiv metadata + ogbn-arxiv citation edges
# ----------------------------------------------------------------------------------
#
# Everything here is REAL, so everything planted is gone: no true_support, and B1
# therefore loses `planted` -- the yardstick that lets it grade a label source at all.
# That is exactly the README's argument for a synthetic corpus, and it is the honest
# price of real data.
#
# What real data buys instead is the two things hardest to fake:
#   * genuine version chains -- ~38% of arXiv papers have more than one version,
#     against 2.7% of the synthetic corpus, so the supersession hard check and
#     hybrid_rrf_fresh finally get a real workout
#   * genuine citation edges, carrying the real incompleteness B1 is about
# ----------------------------------------------------------------------------------

DATA_DIR = STATE_DIR / "data"
_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", (s or "").replace("\n", " ")).strip()


def _snowball(adj: dict, seeds: Sequence[str], target: int,
              n_versions: dict) -> list[str]:
    """Grow a connected subgraph until it holds `target` CARDS (versions, not papers).

    Random sampling would be wrong here rather than merely worse: drawing 12k of 158k
    papers independently retains only ~(12/158)^2 of the edges, so the citation graph
    -- the whole reason for using this dataset -- would essentially vanish.
    """
    seen, order, frontier = set(seeds), list(seeds), list(seeds)
    cards = sum(n_versions.get(p, 1) for p in seeds)
    while frontier and cards < target:
        nxt = []
        for p in frontier:
            for q in adj.get(p, ()):
                if q in seen:
                    continue
                seen.add(q); order.append(q); nxt.append(q)
                cards += n_versions.get(q, 1)
                if cards >= target:
                    return order
        frontier = nxt
    return order


def version_cited_at(chain: Sequence[tuple[datetime, str]], when: datetime) -> Optional[str]:
    """The version of a cited paper that existed when the citing card appeared, or None.

    None drops the edge. A paper cited before its first version existed is a citation
    from the future; pointing it at v1 anyway (the old fallback) planted exactly the
    forward-in-time edges the snapshot mask exists to rule out."""
    best = None
    for ts, cid in chain:
        if ts <= when:
            best = cid
    return best


def build_corpus_arxiv(cfg: Config, n_cards: int) -> Corpus:
    meta_p = DATA_DIR / "arxiv-subset.jsonl"
    cites_p = DATA_DIR / "citations-arxiv.tsv"
    for f in (meta_p, cites_p):
        if not f.exists():
            raise SystemExit(f"missing {f}; fetch the arXiv data first")

    papers: dict = {}
    dropped: dict = defaultdict(int)
    with meta_p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                dropped["unparseable"] += 1; continue
            title, abstract = _clean(d.get("title")), _clean(d.get("abstract"))
            if len(abstract) < 200 or len(title) < 10:
                dropped["too_short"] += 1; continue
            vs = {}
            for v in d.get("versions") or []:
                try:
                    vs[int(str(v["version"]).lstrip("v"))] = parsedate_to_datetime(
                        v["created"])
                except Exception:
                    continue
            if not vs:
                dropped["no_usable_version"] += 1; continue
            cats = (d.get("categories") or "").split()
            if not cats:
                dropped["no_category"] += 1; continue
            papers[d["id"]] = dict(title=title, abstract=abstract, cat=cats[0],
                                   versions=sorted(vs.items()))
    log(f"loaded {len(papers)} papers  dropped={dict(dropped)}", indent=1)

    cites: dict = defaultdict(list)
    adj: dict = defaultdict(set)
    with cites_p.open() as f:
        for line in f:
            a, _, b = line.partition("\t")
            b = b.strip()
            if a in papers and b in papers and a != b:
                cites[a].append(b)
                adj[a].add(b); adj[b].add(a)

    nver = {pid: len(p["versions"]) for pid, p in papers.items()}
    rng = random.Random(cfg.seed)
    hubs = sorted(adj, key=lambda x: -len(adj[x]))[:400]
    seeds = rng.sample(hubs or list(papers), k=min(40, len(hubs or papers)))
    keep_order = _snowball(adj, seeds, n_cards, nver)
    keep = set(keep_order)
    outside = [p for p in papers if p not in keep]
    log(f"snowball: {len(keep)} papers -> {sum(nver[p] for p in keep)} cards", indent=1)

    cats = sorted({papers[p]["cat"] for p in keep})
    cat_ix = {c: i for i, c in enumerate(cats)}
    cards: list = []
    chains: dict = {}
    for pid in keep:
        d = papers[pid]
        one_line = _clean(d["abstract"].split(". ")[0])[:300]
        prev = None
        chain = []
        for n, ts in d["versions"]:
            cid = f"{pid}v{n}"
            cards.append(Card(
                id=cid, proofline_id=d["cat"], topic=cat_ix[d["cat"]],
                kc_type="literature_note", kc_stage="lit_review",
                title=d["title"], one_line=one_line, body=d["abstract"],
                committed_at=ts, version=n, supersedes_id=prev, is_current=False,
                parent_ids=[], citation_ids=[], true_support=[],
                terms=[], inherited_terms=[], findability=0.0))
            chain.append((ts, cid)); prev = cid
        cards[-1].is_current = True
        chains[pid] = chain

    by_id = {c.id: c for c in cards}

    n_edges = 0
    for c in cards:
        pid = c.id.rsplit("v", 1)[0]
        for tgt in cites.get(pid, ()):
            if tgt in keep:
                t = version_cited_at(chains.get(tgt, ()), c.committed_at)
                if t and t != c.id:
                    c.citation_ids.append(t); n_edges += 1

    queries: dict = defaultdict(list)
    current = [c for c in cards if c.is_current]
    # One query per current card is ~32k at the 60k-card point: far past the n needed
    # for power, and the dominant cost in every branch.
    q_pool = current if len(current) <= cfg.max_eval_queries else rng.sample(
        current, k=cfg.max_eval_queries)

    for i, c in enumerate(q_pool):
        rel = {t: 2.0 for t in c.citation_ids
               if t in by_id and by_id[t].committed_at < c.committed_at}
        if not rel:
            continue
        toks = c.title.split()
        cross = len({by_id[k].proofline_id for k in rel}) > 1
        recent = any((c.committed_at - by_id[k].committed_at).days <= 30 for k in rel)
        queries["dag_mined"].append(Query(
            id=f"q_dg_{i:06d}", text=c.title, provenance="dag_mined",
            as_of=c.committed_at, source_card=c.id, rel=rel,
            slices=(["short" if len(toks) < 8 else "long"]
                    + (["cross_project"] if cross else [])
                    + (["recently_changed"] if recent else [])),
            answerable=True))

    for i, c in enumerate(q_pool):
        words = c.body.split()
        if len(words) < 12:
            continue
        queries["generated"].append(Query(
            id=f"q_gn_{i:06d}", text=" ".join(rng.sample(words, k=9)),
            provenance="generated", as_of=c.committed_at, source_card=c.id,
            rel={c.id: 2.0},
            slices=["long"] + (["recently_changed"] if c.version > 1 else []),
            answerable=True))

    # NO-ANSWER from papers held OUT of the corpus entirely: the evidence genuinely is
    # not there. Subsetting supplies these for free -- nothing has to be withheld.
    # One no-answer query per style. They must match the shape of the answerable
    # queries they sit beside: a title-shaped no-answer among abstract-word queries
    # would be separable on form alone, and the abstention threshold would be
    # measuring query style rather than whether the answer is in the corpus.
    latest_ts = max(c.committed_at for c in cards)
    for i, pid in enumerate(rng.sample(outside,
                                       k=min(len(outside), max(40, len(q_pool) // 5)))):
        t = papers[pid]["title"].split()
        queries["dag_mined"].append(Query(
            id=f"q_na_dg_{i:06d}", text=" ".join(t[:10]), provenance="dag_mined",
            as_of=latest_ts, source_card=None, rel={},
            slices=["no_answer", "short" if len(t) < 8 else "long"], answerable=False))
        w = papers[pid]["abstract"].split()
        if len(w) >= 12:
            queries["generated"].append(Query(
                id=f"q_na_gn_{i:06d}", text=" ".join(rng.sample(w, k=9)),
                provenance="generated", as_of=latest_ts, source_card=None, rel={},
                slices=["no_answer", "long"], answerable=False))

    topics: dict = defaultdict(list)
    for c in cards:
        topics[c.topic].append(c.id)
    n_ans = sum(1 for q in queries["generated"] if q.answerable)
    n_dag_ans = sum(1 for q in queries["dag_mined"] if q.answerable)
    stats = {
        "source": "arxiv",
        "cards": len(cards),
        "papers": len(keep),
        "current_cards": sum(1 for c in cards if c.is_current),
        "superseded": sum(1 for c in cards if not c.is_current),
        "multi_version_papers": sum(1 for p in keep if nver[p] > 1),
        "prooflines(categories)": len(cats),
        "citation_edges": n_edges,
        "q_dag_mined": n_dag_ans,
        "q_generated": n_ans,
        "q_no_answer": (len(queries["generated"]) - n_ans) + (
            len(queries["dag_mined"]) - n_dag_ans),
    }
    return Corpus(cards=cards, by_id=by_id, queries=dict(queries),
                  held_out_prooflines=set(), topics=dict(topics), stats=stats)


# ----------------------------------------------------------------------------------
# §2  Storage: Postgres
# ----------------------------------------------------------------------------------
#
# Postgres and not Mongo, for three reasons that matter to this problem specifically:
#   * the record is a DAG, so ancestry and provenance closure are recursive CTEs
#   * "the corpus as of commit X" is a range predicate, and it has to be cheap
#   * tsvector/GIN gives a real lexical scorer with no extra dependency
#
# And deliberately NO pgvector. At a few thousand to tens of thousands of pieces,
# brute-force cosine over a float32 matrix is single-digit milliseconds, faster than
# an unindexed pgvector scan and with no install step. ANN indexing is a problem this
# product does not have yet, and pretending otherwise costs a dependency and a lie.
# ----------------------------------------------------------------------------------

import psycopg
from psycopg.rows import dict_row

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
  id            text PRIMARY KEY,
  proofline_id  text NOT NULL,
  topic         int  NOT NULL,
  kc_type       text NOT NULL,
  kc_stage      text NOT NULL,
  title         text NOT NULL,
  one_line      text NOT NULL,
  body          text NOT NULL,
  committed_at  timestamptz NOT NULL,
  version       int  NOT NULL,
  supersedes_id text REFERENCES cards(id) DEFERRABLE INITIALLY DEFERRED,
  is_current    boolean NOT NULL,
  txt           text NOT NULL,
  fts           tsvector GENERATED ALWAYS AS (to_tsvector('simple', txt)) STORED,
  embedding     bytea,
  ndim          int
);
CREATE INDEX IF NOT EXISTS cards_fts    ON cards USING gin (fts);
CREATE INDEX IF NOT EXISTS cards_asof   ON cards (committed_at);
CREATE INDEX IF NOT EXISTS cards_pl     ON cards (proofline_id);

-- edges: structural (author-recorded) vs planted truth, kept apart on purpose.
-- Only 'parent' and 'citation' are observable in a real deployment. 'true_support'
-- exists here so the harness can grade its own labels; nothing that runs in
-- production is allowed to read it. See assert_no_truth_leak().
CREATE TABLE IF NOT EXISTS edges (
  src        text NOT NULL,
  dst        text NOT NULL,
  kind       text NOT NULL CHECK (kind IN ('parent','citation','true_support')),
  valid_from timestamptz NOT NULL,       -- the citing card's committed_at
  PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS edges_dst ON edges (dst, kind);

-- PLANTED entity layer. Truth, like true_support: B1 grades extractors against it and
-- nothing on the retrieval path may read it (assert_no_truth_leak).
CREATE TABLE IF NOT EXISTS true_entities (
  id text PRIMARY KEY, name text NOT NULL, kind text NOT NULL, parent_kind text);
CREATE TABLE IF NOT EXISTS true_mentions (
  card_id text NOT NULL, entity_id text NOT NULL, surface text NOT NULL, role text NOT NULL);
CREATE TABLE IF NOT EXISTS true_facts (
  card_id text NOT NULL, subj text NOT NULL, rel text NOT NULL, obj text NOT NULL, value text);

-- EXTRACTED, written by ingest_semantica.py. `source` names the extractor
-- ('semantica:regex', 'semantica:llm:anthropic:claude-haiku-4-5'); several coexist.
CREATE TABLE IF NOT EXISTS entities (
  id text NOT NULL, name text NOT NULL, kind text NOT NULL, source text NOT NULL,
  PRIMARY KEY (id, source));
CREATE TABLE IF NOT EXISTS mentions (
  card_id text NOT NULL, entity_id text NOT NULL, surface text NOT NULL,
  role text NOT NULL, source text NOT NULL);
CREATE INDEX IF NOT EXISTS mentions_card ON mentions (source, card_id);
CREATE TABLE IF NOT EXISTS facts (
  card_id text NOT NULL, subj text NOT NULL, rel text NOT NULL, obj text NOT NULL,
  value text, source text NOT NULL,
  valid_from timestamptz NOT NULL, valid_until timestamptz);    -- NULL = still valid
CREATE TABLE IF NOT EXISTS onto_classes (source text NOT NULL, class text NOT NULL, parent text);
-- which cards an extractor actually read: B1 grades only those
CREATE TABLE IF NOT EXISTS extracted_cards (
  card_id text NOT NULL, source text NOT NULL, PRIMARY KEY (card_id, source));

CREATE TABLE IF NOT EXISTS queries (
  id          text PRIMARY KEY,
  provenance  text NOT NULL,
  text        text NOT NULL,
  as_of       timestamptz NOT NULL,
  source_card text,
  answerable  boolean NOT NULL,
  slices      text[] NOT NULL,
  intent      text NOT NULL DEFAULT 'current',   -- current | history (declared, like as_of)
  family      text,                              -- multihop family
  answer      jsonb,                             -- gold answer (P3 scores it)
  pair_id     text,                              -- contrast group
  dropped     text                               -- NULL = kept | single_hop | pair_dropped
);

CREATE TABLE IF NOT EXISTS qrels (
  query_id text NOT NULL,
  card_id  text NOT NULL,
  grade    real NOT NULL,
  slot     smallint,                             -- hop index (multihop only)
  PRIMARY KEY (query_id, card_id)
);

CREATE TABLE IF NOT EXISTS runs (
  id         text PRIMARY KEY,
  scorer     text NOT NULL,
  label_src  text NOT NULL,
  corpus_n   int  NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  meta       jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS results (
  run_id   text NOT NULL,
  query_id text NOT NULL,
  rank     int  NOT NULL,
  card_id  text NOT NULL,
  score    real NOT NULL,
  PRIMARY KEY (run_id, query_id, rank)
);

CREATE TABLE IF NOT EXISTS metrics (
  run_id   text NOT NULL,
  query_id text NOT NULL,
  name     text NOT NULL,
  value    real NOT NULL,
  PRIMARY KEY (run_id, query_id, name)
);
"""


def time_valid_edges(corpus: Corpus) -> tuple[list[tuple[str, str, str, datetime]], dict[str, int]]:
    """Every edge whose target existed when its source was written, with that time.

    An edge that points forward in time is not a citation anyone could have made, and a
    graph feature built on it scores a 1998 query with 2020 structure. Checked per edge,
    so a successor that inherited a parent list keeps whatever is valid for IT."""
    rows: list[tuple[str, str, str, datetime]] = []
    dropped: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    for c in corpus.cards:
        for kind, lst in (("parent", c.parent_ids), ("citation", c.citation_ids),
                          ("true_support", c.true_support)):
            for d in lst:
                if (c.id, d, kind) in seen:
                    continue
                seen.add((c.id, d, kind))
                dst = corpus.by_id.get(d)
                if dst is None or dst.committed_at > c.committed_at:
                    dropped[kind] += 1
                    continue
                rows.append((c.id, d, kind, c.committed_at))
    return rows, dict(dropped)


class Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._pgserver = None
        url = cfg.database_url or os.environ.get("DATABASE_URL")
        if not url:
            url = self._boot_embedded()
        self.url = url
        self.conn = psycopg.connect(url, row_factory=dict_row, autocommit=True)

    def _boot_embedded(self) -> str:
        try:
            import pgserver
        except ImportError:
            sys.exit(
                "No DATABASE_URL, and the embedded Postgres (pgserver) is not available\n"
                f"on this platform ({sys.platform}/{os.uname().machine if hasattr(os,'uname') else '?'}).\n"
                "pgserver ships wheels for macOS (x86_64 and arm64), Linux x86_64 and\n"
                "Windows x86_64 only. Point the harness at any Postgres 14+ instead:\n\n"
                "  export DATABASE_URL=postgresql://localhost/proofline\n\n"
                "  # or, if you have Docker:\n"
                "  docker run -d --name pl -e POSTGRES_PASSWORD=pl -p 5432:5432 postgres:16\n"
                "  export DATABASE_URL=postgresql://postgres:pl@localhost:5432/postgres\n")
        d = STATE_DIR / "pgdata"
        d.mkdir(parents=True, exist_ok=True)
        log(f"booting embedded postgres in {d} ...")
        self._pgserver = pgserver.get_server(str(d), cleanup_mode=None)
        return self._pgserver.get_uri()

    def init(self, *, reset: bool = False) -> None:
        with self.conn.cursor() as cur:
            if reset:
                cur.execute("DROP TABLE IF EXISTS metrics, results, runs, qrels, queries, "
                            "edges, cards, true_entities, true_mentions, true_facts, "
                            "entities, mentions, facts, onto_classes, extracted_cards CASCADE;")
            cur.execute(SCHEMA)

    def load_corpus(self, corpus: Corpus) -> None:
        with self.conn.cursor() as cur:
            cur.execute("SET session_replication_role = replica;")  # defer FK during bulk
            with cur.copy(
                "COPY cards (id,proofline_id,topic,kc_type,kc_stage,title,one_line,"
                "body,committed_at,version,supersedes_id,is_current,txt) FROM STDIN"
            ) as cp:
                for c in corpus.cards:
                    cp.write_row((c.id, c.proofline_id, c.topic, c.kc_type, c.kc_stage,
                                  c.title, c.one_line, c.body, c.committed_at, c.version,
                                  c.supersedes_id, c.is_current, c.text))
            edge_rows, dropped = time_valid_edges(corpus)
            corpus.stats["forward_edges_dropped"] = dropped
            with cur.copy("COPY edges (src,dst,kind,valid_from) FROM STDIN") as cp:
                for r in edge_rows:
                    cp.write_row(r)
            with cur.copy("COPY true_entities (id,name,kind,parent_kind) FROM STDIN") as cp:
                for e in corpus.entities:
                    cp.write_row(e)
            with cur.copy("COPY true_mentions (card_id,entity_id,surface,role) FROM STDIN") as cp:
                for c in corpus.cards:
                    for eid, surface, role in c.true_mentions:
                        cp.write_row((c.id, eid, surface, role))
            with cur.copy("COPY true_facts (card_id,subj,rel,obj,value) FROM STDIN") as cp:
                for c in corpus.cards:
                    for s, r, o, v in c.true_facts:
                        cp.write_row((c.id, s, r, o, v))
            with cur.copy(
                "COPY queries (id,provenance,text,as_of,source_card,answerable,slices,"
                "intent,family,answer,pair_id) FROM STDIN"
            ) as cp:
                for prov, qs in corpus.queries.items():
                    for q in qs:
                        cp.write_row((q.id, prov, q.text, q.as_of, q.source_card,
                                      q.answerable, q.slices, q.intent, q.family,
                                      json.dumps(q.answer) if q.answer is not None else None,
                                      q.pair_id))
            with cur.copy("COPY qrels (query_id,card_id,grade,slot) FROM STDIN") as cp:
                for qs in corpus.queries.values():
                    for q in qs:
                        if q.slots:
                            for i, members in enumerate(q.slots):
                                for cid in sorted(members):
                                    cp.write_row((q.id, cid, q.rel.get(cid, 2.0), i))
                        else:
                            for cid, g in q.rel.items():
                                cp.write_row((q.id, cid, g, None))
            cur.execute("SET session_replication_role = origin;")
            cur.execute("ANALYZE;")

    def set_embeddings(self, ids: Sequence[str], mat: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE _emb (id text, e bytea, n int) ON COMMIT DROP;"
                        if False else
                        "CREATE TEMP TABLE IF NOT EXISTS _emb (id text, e bytea, n int);")
            cur.execute("TRUNCATE _emb;")
            with cur.copy("COPY _emb (id,e,n) FROM STDIN") as cp:
                for i, cid in enumerate(ids):
                    cp.write_row((cid, mat[i].astype(np.float32).tobytes(), mat.shape[1]))
            cur.execute("UPDATE cards c SET embedding = t.e, ndim = t.n "
                        "FROM _emb t WHERE t.id = c.id;")

    def snapshot(self, as_of: datetime, n_limit: Optional[int] = None) -> dict:
        """The corpus as it stood at `as_of`. Everything downstream must go through
        this: a frozen test set evaluated against today's corpus quietly penalises the
        scorer that finds the current version."""
        q = ("SELECT id, proofline_id, txt, embedding, is_current, supersedes_id, "
             "committed_at FROM cards WHERE committed_at <= %s ORDER BY committed_at")
        with self.conn.cursor() as cur:
            cur.execute(q, (as_of,))
            rows = cur.fetchall()
        if n_limit:
            rows = rows[:n_limit]
        return {r["id"]: r for r in rows}

    def subset(self, keep: Optional[frozenset], n_total: Optional[int]) -> dict:
        """Rows for a scale point.

        A scale test that truncates the record also deletes the answers, and then
        reports that retrieval got *better* as the corpus grew, because the queries it
        could not answer left the set. The needle stays; only the haystack grows.
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, proofline_id, txt, embedding, ndim, is_current, "
                        "supersedes_id, committed_at FROM cards ORDER BY id")
            rows = cur.fetchall()
        if n_total is None or n_total >= len(rows):
            return {r["id"]: r for r in rows}
        keep = keep or frozenset()
        needles = [r for r in rows if r["id"] in keep]
        hay = [r for r in rows if r["id"] not in keep]
        room = max(0, n_total - len(needles))
        rng = random.Random(1234)
        rng.shuffle(hay)
        return {r["id"]: r for r in needles + hay[:room]}

    def close(self) -> None:
        try:
            self.conn.close()
        finally:
            if self._pgserver is not None:
                pass  # leave the server up between commands; `reset` cleans the data


# ----------------------------------------------------------------------------------
# §3  Embeddings
# ----------------------------------------------------------------------------------

class Embedder:
    name = "base"
    dim = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode DOCUMENTS."""
        raise NotImplementedError

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode QUERIES. Symmetric by default; a two-tower model overrides this.

        Retrieval quality depends on the two sides being encoded the way the model
        was trained, and a model that wants separate towers will still return
        plausible vectors from the wrong one -- so the failure is silent."""
        return self.encode(texts)


class HashEmbedder(Embedder):
    """Signed random projection of word + char-3gram hashes. No download, no key,
    deterministic. It is a *lexical-ish* vector, not a semantic one, good enough to
    exercise every code path offline, and honest about what it is. Use
    sentence-transformers for a real dense scorer."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams = toks + [t[i:i + 3] for t in toks for i in range(max(1, len(t) - 2))]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.m = SentenceTransformer(model)
        self.dim = self.m.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True,
                                        batch_size=64, show_progress_bar=False),
                          dtype=np.float32)


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        import httpx
        self.model, self.dim = model, 1536
        self.key = os.environ["OPENAI_API_KEY"]
        self.http = httpx.Client(timeout=90)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            r = self.http.post("https://api.openai.com/v1/embeddings",
                               headers={"Authorization": f"Bearer {self.key}"},
                               json={"model": self.model, "input": list(texts[i:i + 256])})
            r.raise_for_status()
            out += [d["embedding"] for d in r.json()["data"]]
        m = np.asarray(out, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


class Specter2Embedder(Embedder):
    """SPECTER2 (allenai) -- a scientific-document encoder, and an ASYMMETRIC one.

    Per the model card, a short free-text query is encoded with the `adhoc_query`
    adapter while candidate documents use `proximity`. Using one adapter for both
    sides is the standard way to get quietly mediocre retrieval out of this model:
    it still returns unit vectors and plausible-looking scores.

    Note what it is being asked to do here. SPECTER2 was trained on real titles and
    abstracts; this corpus is machine-generated vocabulary, so its subword tokeniser
    sees nonsense and the embeddings carry far less meaning than they would on a real
    record. It is the right encoder for the AMiner/arXiv loader the README describes,
    not obviously the right one for the synthetic corpus.
    """

    name = "specter2"
    BASE = "allenai/specter2_base"
    DOC_ADAPTER = "allenai/specter2"                  # the proximity/retrieval adapter
    QUERY_ADAPTER = "allenai/specter2_adhoc_query"

    def __init__(self, batch: int = 16):
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel
        self.torch, self.batch = torch, batch
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.BASE)
        self.m = AutoAdapterModel.from_pretrained(self.BASE)
        self.m.load_adapter(self.DOC_ADAPTER, source="hf", load_as="proximity")
        self.m.load_adapter(self.QUERY_ADAPTER, source="hf", load_as="adhoc_query")
        self.m.to(self.device).eval()
        self.dim = int(self.m.config.hidden_size)
        log(f"  specter2: device={self.device} dim={self.dim} "
            f"adapters=proximity(docs)+adhoc_query(queries)")

    def _encode(self, texts: Sequence[str], adapter: str) -> np.ndarray:
        self.m.set_active_adapters(adapter)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch):
                b = self.tok(list(texts[i:i + self.batch]), padding=True,
                             truncation=True, return_tensors="pt",
                             return_token_type_ids=False, max_length=512)
                b = {k: v.to(self.device) for k, v in b.items()}
                # CLS token, as the model card specifies.
                h = self.m(**b).last_hidden_state[:, 0, :]
                out.append(h.float().cpu().numpy())
        m = np.vstack(out).astype(np.float32)
        # Index.cosine() is a bare dot product, so the vectors must carry their own
        # normalisation or "cosine similarity" silently becomes an inner product.
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # txt is "title . one_line . body"; SPECTER2 was trained on
        # title <sep> abstract, so restore the separator on the first boundary.
        sep = self.tok.sep_token or " "
        return self._encode([t.replace(" . ", sep, 1) for t in texts], "proximity")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), "adhoc_query")


def make_embedder(cfg: Config) -> Embedder:
    if cfg.embedder == "hash":
        return HashEmbedder(cfg.embed_dim)
    if cfg.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if cfg.embedder == "openai":
        return OpenAIEmbedder()
    if cfg.embedder == "specter2":
        return Specter2Embedder()
    raise SystemExit(f"unknown embedder {cfg.embedder}")

# ----------------------------------------------------------------------------------
# §4  Retrieval index (built per corpus snapshot)
# ----------------------------------------------------------------------------------

class Index:
    """Everything a scorer may look at, for one snapshot of the record.

    `graph` is present but a scorer must declare `uses_graph = True` to touch it, and
    declaring it makes the harness refuse to evaluate that scorer against labels that
    were themselves derived from the graph. See check_circularity().
    """

    def __init__(self, rows: dict, k1: float = 1.2, b: float = 0.75):
        self.ids = list(rows.keys())
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.rows = rows
        self.k1, self.b = k1, b

        self.docs = [re.findall(r"[a-z0-9]+", rows[c]["txt"].lower()) for c in self.ids]
        self.dl = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if len(self.dl) else 1.0

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(self.docs):
            tf: dict[str, int] = defaultdict(int)
            for t in d:
                tf[t] += 1
            for t, f in tf.items():
                self.postings[t].append((i, f))
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

        embs = [rows[c]["embedding"] for c in self.ids]
        if all(e is not None for e in embs) and embs:
            d = rows[self.ids[0]]["ndim"] if "ndim" in rows[self.ids[0]] else None
            arr = [np.frombuffer(e, dtype=np.float32) for e in embs]
            self.emb = np.vstack(arr)
        else:
            self.emb = None

        self.superseded_by: dict[str, str] = {}
        for cid, r in rows.items():
            if r["supersedes_id"] and r["supersedes_id"] in rows:
                self.superseded_by[r["supersedes_id"]] = cid
        self.is_current = {cid: bool(r["is_current"]) for cid, r in rows.items()}
        self.current_mask = np.array([bool(rows[c]["is_current"]) for c in self.ids])
        # When each card stopped being current: its successor's commit, or never. The
        # end-of-record `current_mask` answers "current today", which no query may ask.
        self.superseded_ts = np.full(len(self.ids), np.inf)
        for cid, r in rows.items():
            sup = r["supersedes_id"]
            if sup and sup in self.pos:
                j = self.pos[sup]
                self.superseded_ts[j] = min(self.superseded_ts[j], r["committed_at"].timestamp())
        self.ts = np.array([rows[c]["committed_at"].timestamp() for c in self.ids],
                           dtype=np.float64)
        self._mask_cache: dict[float, np.ndarray] = {}
        self.graph: dict[str, list[str]] = {}
        self._cite_ts: dict[str, np.ndarray] = {}

    def attach_graph(self, edges: Sequence[tuple[str, str, datetime]]) -> None:
        """`edges` are (src, dst, valid_from). In-degree is counted AS OF a query time: a
        citation written after `as_of` did not exist yet, and one from a card outside
        this corpus (a scale point's sample) does not exist at all."""
        g: dict[str, list[str]] = defaultdict(list)
        cites: dict[str, list[float]] = defaultdict(list)
        for src, dst, vf in edges:
            g[src].append(dst)
            if dst in self.pos and src in self.pos:
                cites[dst].append(vf.timestamp())
        self.graph = dict(g)
        self._cite_ts = {d: np.sort(np.asarray(ts, dtype=np.float64)) for d, ts in cites.items()}

    def indeg_at(self, cid: str, as_of: datetime) -> int:
        ts = self._cite_ts.get(cid)
        if ts is None:
            return 0
        return int(np.searchsorted(ts, as_of.timestamp(), side="right"))

    def snapshot_mask(self, as_of: datetime) -> np.ndarray:
        """The corpus as it stood at `as_of`, as a boolean mask.

        Masking beats rebuilding: one index, one posting list, and every query still
        sees only what existed when the need arose. Rebuilding per timestamp is what
        makes a snapshot-correct harness too slow to run on every change, and a harness
        that is too slow to run on every change does not get run.
        """
        k = as_of.timestamp()
        m = self._mask_cache.get(k)
        if m is None:
            m = self.ts <= k
            if len(self._mask_cache) > 4096:
                self._mask_cache.clear()
            self._mask_cache[k] = m
        return m

    def current_at(self, as_of: datetime) -> np.ndarray:
        """Cards not yet superseded at `as_of`. A card revised AFTER the query's date was
        the current version when the question was asked; filtering it on today's
        `current_mask` drops it while its successor is still masked out as unwritten."""
        return self.superseded_ts > as_of.timestamp()

    def age_days_at(self, as_of: datetime) -> np.ndarray:
        return np.maximum(0.0, (as_of.timestamp() - self.ts) / 86400.0)

    # -- primitives ----------------------------------------------------------------
    def bm25(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.ids), dtype=np.float32)
        qt = re.findall(r"[a-z0-9]+", query.lower())
        for t in qt:
            p = self.postings.get(t)
            if not p:
                continue
            idf = self.idf[t]
            for i, f in p:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s[i] += idf * (f * (self.k1 + 1)) / denom
        return s

    def cosine(self, qvec: np.ndarray) -> np.ndarray:
        if self.emb is None:
            return np.zeros(len(self.ids), dtype=np.float32)
        return self.emb @ qvec.astype(np.float32)


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for r in rankings:
        for rank, idx in enumerate(r, start=1):
            out[idx] += 1.0 / (k + rank)
    return out


def topn(scores: np.ndarray, n: int) -> list[int]:
    if n >= len(scores):
        return list(np.argsort(-scores))
    part = np.argpartition(-scores, n)[:n]
    return list(part[np.argsort(-scores[part])])


# ----------------------------------------------------------------------------------
# §5  Scorers
# ----------------------------------------------------------------------------------

@dataclass
class Scored:
    ids: list[str]
    scores: list[float]
    top_score: float          # used for the abstain decision (branch 3)


class Scorer:
    name = "base"
    uses_graph = False
    description = ""

    def prepare(self, idx: Index, embedder: Embedder) -> None:
        self.idx, self.embedder = idx, embedder

    def run(self, q: Query, k: int) -> Scored:
        raise NotImplementedError

    @staticmethod
    def _mask(scores: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = np.where(m, scores, -np.inf)
        return out

    def _emit(self, order: list[int], scores: np.ndarray, k: int) -> Scored:
        pairs = [(self.idx.ids[i], float(scores[i])) for i in order[:k]
                 if np.isfinite(scores[i]) and scores[i] > 0]
        ids = [x for x, _ in pairs]
        sc = [y for _, y in pairs]
        return Scored(ids, sc, sc[0] if sc else 0.0)


class BM25Scorer(Scorer):
    name = "bm25"
    description = "Okapi BM25, k1=1.2 b=0.75. The incumbent."

    def run(self, q, k):
        s = self._mask(self.idx.bm25(q.text), self.idx.snapshot_mask(q.as_of))
        return self._emit(topn(s, k), s, k)


class DenseScorer(Scorer):
    name = "dense"
    description = "Cosine over embeddings, brute force. No ANN index at this scale."

    def run(self, q, k):
        v = self.embedder.encode_queries([q.text])[0]
        s = self._mask(self.idx.cosine(v), self.idx.snapshot_mask(q.as_of))
        return self._emit(topn(s, k), s, k)


class HybridRRF(Scorer):
    name = "hybrid_rrf"
    description = "Reciprocal rank fusion of BM25 and dense, k=60."

    def run(self, q, k):
        m = self.idx.snapshot_mask(q.as_of)
        lex = self._mask(self.idx.bm25(q.text), m)
        den = self._mask(self.idx.cosine(self.embedder.encode_queries([q.text])[0]), m)
        fused = rrf([topn(lex, 100), topn(den, 100)])
        s = np.zeros(len(self.idx.ids), dtype=np.float32)
        for i, v in fused.items():
            if m[i]:
                s[i] = v
        return self._emit(topn(s, k), s, k)


class HybridFresh(Scorer):
    """RRF plus two metadata rules, and this is the argument the file is making:

    a superseded card and its successor are near-identical in *meaning*, so no semantic
    scorer can separate them. The separation is metadata. Doing it in the candidate set
    turns a ranking trade-off into a hard guarantee, which is why the superseded check
    below can be a hard zero rather than a number in a table.
    """
    name = "hybrid_rrf_fresh"
    description = "RRF + drop superseded + mild recency decay."

    def run(self, q, k):
        m = self.idx.snapshot_mask(q.as_of)
        lex = self._mask(self.idx.bm25(q.text), m)
        den = self._mask(self.idx.cosine(self.embedder.encode_queries([q.text])[0]), m)
        fused = rrf([topn(lex, 150), topn(den, 150)])
        age = self.idx.age_days_at(q.as_of)
        cur = self.idx.current_at(q.as_of)
        s = np.zeros(len(self.idx.ids), dtype=np.float32)
        for i, v in fused.items():
            if not m[i] or not cur[i]:
                continue                       # filtered, not down-weighted
            s[i] = v / (1.0 + 0.0006 * float(age[i]))
        return self._emit(topn(s, k), s, k)


class GraphBoost(Scorer):
    """Deliberately circular against DAG-mined labels. It exists so the harness has
    something to refuse; a guard nobody ever trips is a guard nobody trusts."""
    name = "graph_boost"
    uses_graph = True
    description = "RRF + in-degree boost from the record's own link structure."

    def run(self, q, k):
        m = self.idx.snapshot_mask(q.as_of)
        lex = self._mask(self.idx.bm25(q.text), m)
        den = self._mask(self.idx.cosine(self.embedder.encode_queries([q.text])[0]), m)
        fused = rrf([topn(lex, 150), topn(den, 150)])
        s = np.zeros(len(self.idx.ids), dtype=np.float32)
        for i, v in fused.items():
            if not m[i]:
                continue
            s[i] = v * (1.0 + 0.35 * math.log1p(self.idx.indeg_at(self.idx.ids[i], q.as_of)))
        return self._emit(topn(s, k), s, k)


_CE_CACHE: dict = {}


class CrossEncoderRerank(Scorer):
    """hybrid_rrf_fresh for candidates, then a cross-encoder over the top 50.

    The brief claims this is probably the largest single win available, and that if
    you have not tried it, the gap between two first-stage scorers may not be the most
    interesting comparison on the table. That is testable, so it is a scorer.
    """

    name = "rerank"
    description = "hybrid_rrf_fresh top-50, reranked by a cross-encoder."
    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CAND = 50

    def prepare(self, idx, embedder):
        super().prepare(idx, embedder)
        ce = _CE_CACHE.get(self.MODEL)
        if ce is None:
            from sentence_transformers import CrossEncoder
            import torch
            dev = ("mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
            ce = CrossEncoder(self.MODEL, device=dev, max_length=384)
            log(f"  cross-encoder {self.MODEL} on {dev}")
            _CE_CACHE[self.MODEL] = ce
        self.ce = ce

    def run(self, q, k):
        m = self.idx.snapshot_mask(q.as_of)
        lex = self._mask(self.idx.bm25(q.text), m)
        den = self._mask(self.idx.cosine(self.embedder.encode_queries([q.text])[0]), m)
        fused = rrf([topn(lex, 150), topn(den, 150)])
        cur = self.idx.current_at(q.as_of)
        cand = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])
                if m[i] and cur[i]][:self.CAND]
        if not cand:
            return Scored([], [], 0.0)
        pairs = [(q.text, self.idx.rows[self.idx.ids[i]]["txt"][:1200]) for i in cand]
        sc = self.ce.predict(pairs, batch_size=64, show_progress_bar=False)
        order = sorted(range(len(cand)), key=lambda j: -float(sc[j]))[:k]
        ids = [self.idx.ids[cand[j]] for j in order]
        scores = [float(sc[j]) for j in order]
        # Not via _emit: cross-encoder outputs are logits and go negative, and _emit
        # drops anything <= 0.
        return Scored(ids, scores, scores[0] if scores else 0.0)


SCORERS: dict[str, type[Scorer]] = {c.name: c for c in
                                    [BM25Scorer, DenseScorer, HybridRRF,
                                     HybridFresh, GraphBoost, CrossEncoderRerank]}

# label sources that are derived from the record's link structure
GRAPH_DERIVED_LABELS = {"dag_mined"}


def check_circularity(scorer: Scorer, label_src: str) -> Optional[str]:
    if scorer.uses_graph and label_src in GRAPH_DERIVED_LABELS:
        return (f"REFUSED: scorer '{scorer.name}' reads graph structure and labels "
                f"'{label_src}' are derived from it. The comparison would score the "
                f"scorer on its own feature.")
    return None


# ----------------------------------------------------------------------------------
# §6  Metrics
# ----------------------------------------------------------------------------------
#
# The default is condensed-list nDCG, not recall@k. With mined labels the judgment set
# is INCOMPLETE, so an unjudged document is unknown, not irrelevant. recall@k punishes
# the scorer that surfaces something genuinely relevant the author never linked, which
# is precisely the scorer you were hoping to find.
# ----------------------------------------------------------------------------------

def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], rel: dict[str, float], k: int,
              condensed: bool = True, judged: Optional[set[str]] = None) -> float:
    if not rel:
        return float("nan")
    lst = list(ranked)
    if condensed and judged is not None:
        lst = [c for c in lst if c in judged]
    gains = [rel.get(c, 0.0) for c in lst[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idl = dcg(ideal)
    return (dcg(gains) / idl) if idl > 0 else float("nan")


def mrr(ranked: Sequence[str], rel: dict[str, float]) -> float:
    for i, c in enumerate(ranked, 1):
        if rel.get(c, 0) > 0:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: Sequence[str], rel: dict[str, float], k: int) -> float:
    if not rel:
        return float("nan")
    return len(set(ranked[:k]) & set(rel)) / len(rel)


def bpref(ranked: Sequence[str], rel: dict[str, float], pool: set[str]) -> float:
    """Binary preference. Only judged documents count; unjudged are skipped entirely.

    `pool` is the union of top-k across every scorer under comparison, i.e. TREC-style
    pooling. Pool members not in qrels are judged-nonrelevant; everything outside the
    pool is unjudged and invisible to this metric. That is the point: bpref does not
    change when a scorer finds something nobody has looked at.
    """
    R = len(rel)
    if R == 0:
        return float("nan")
    nonrel_seen, total = 0, 0.0
    N = len([c for c in pool if c not in rel])
    denom = min(R, N) if N else R
    for c in ranked:
        if c in rel:
            total += 1.0 - (min(nonrel_seen, denom) / denom if denom else 0.0)
        elif c in pool:
            nonrel_seen += 1
    return total / R


# ----------------------------------------------------------------------------------
# §7  Statistics
# ----------------------------------------------------------------------------------
#
# Everything here is PAIRED. Two independent means over the same queries throw away
# most of the information and inflate the variance by roughly 2x; the whole reason a
# 300-query set can resolve anything at all is that both scorers see the same queries.
# ----------------------------------------------------------------------------------

def bootstrap_ci(deltas: Sequence[float], n: int = 2000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float, float]:
    d = np.asarray([x for x in deltas if not math.isnan(x)], dtype=np.float64)
    if len(d) < 3:
        return (float("nan"),) * 3
    rs = np.random.default_rng(seed)
    idx = rs.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_power(deltas: Sequence[float], target: float,
                 alpha: float = 0.05, power: float = 0.80) -> dict:
    """The question nobody in the proofline asks: can this many queries see the
    difference you say you care about?"""
    d = np.asarray([x for x in deltas if not math.isnan(x)], dtype=np.float64)
    n = len(d)
    if n < 3:
        return dict(n=n, sd=float("nan"), mde=float("nan"), n_required=float("nan"))
    sd = float(d.std(ddof=1))
    z = 1.959963985 + 0.8416212336               # z_{a/2} + z_beta, ~2.80
    mde = z * sd / math.sqrt(n) if n else float("nan")
    need = (z * sd / target) ** 2 if target > 0 else float("nan")
    return dict(n=n, sd=sd, mde=mde, n_required=math.ceil(need))


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    b01 = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    b10 = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b01 + b10
    if n == 0:
        return dict(b01=0, b10=0, stat=float("nan"), p=float("nan"))
    stat = (abs(b01 - b10) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))
    return dict(b01=b01, b10=b10, stat=stat, p=p)


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pe = sum((sum(1 for x, _ in pairs if x == c) / n) *
             (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    tot = con + dis
    return (con - dis) / tot if tot else float("nan")

# ----------------------------------------------------------------------------------
# §8  LLM judge
# ----------------------------------------------------------------------------------
#
# The judge is treated as an INSTRUMENT, not an oracle. Every judge here is:
#   * pairwise, never absolute: it compares two ranked lists, it does not score one
#   * run twice with the order swapped, so position bias is measured on every item
#   * calibrated against a reference set, and reported with its kappa
# A judge whose kappa against the reference is not stated is a number with no units.
# ----------------------------------------------------------------------------------

JUDGE_PROMPT = """You are comparing two ranked lists of research notes retrieved for a query.

QUERY: {query}

LIST A:
{a}

LIST B:
{b}

Which list better answers the query? Consider whether the top items actually address
the query, and whether obviously redundant or outdated items crowd out useful ones.

Answer with exactly one word: A, B, or TIE."""


Item = tuple[str, str]   # (card_id, rendered text), judges see one or the other


class Judge:
    name = "base"
    tracer: Optional["Tracer"] = None      # set by the branch that builds the judge

    def compare(self, query: str, a: list[Item], b: list[Item]) -> Optional[str]:
        raise NotImplementedError


class MockJudge(Judge):
    """A simulator, not a model. It knows the planted truth, then deliberately corrupts
    its own verdict with the biases the literature reports for real judges, so that the
    bias-measurement code has something to detect when you run offline. Never use its
    numbers as evidence about a real judge."""

    name = "mock"

    def __init__(self, truth: dict[str, dict[str, float]], seed: int = 3,
                 position_bias: float = 0.10, length_bias: float = 0.06):
        self.truth, self.rng = truth, random.Random(seed)
        self.position_bias, self.length_bias = position_bias, length_bias
        self._qid: Optional[str] = None

    def bind(self, qid: str) -> None:
        self._qid = qid

    def compare(self, query, a, b):
        rel = self.truth.get(self._qid or "", {})
        idl = dcg(sorted(rel.values(), reverse=True)[:max(len(a), len(b), 1)]) or 1.0
        sa = sum(rel.get(c[0], 0) / math.log2(i + 2) for i, c in enumerate(a)) / idl
        sb = sum(rel.get(c[0], 0) / math.log2(i + 2) for i, c in enumerate(b)) / idl
        margin = 4.0 * (sa - sb)
        # Position bias attaches to the SLOT, not to the content: whatever is shown
        # first gets a bonus. That is what makes it visible under an order swap, and
        # invisible if you only ever ask once.
        margin += self.position_bias
        # Length bias: prefer the longer list.
        margin += self.length_bias * (len(a) - len(b)) / max(1, max(len(a), len(b)))
        # Deterministic per (query, content) so a swap re-runs the same judge, not a
        # fresh coin flip. Judge noise is real, but it is not the same thing as bias.
        h = hashlib.blake2b((str(self._qid) + "|".join(x[0] for x in a) + "#" +
                             "|".join(x[0] for x in b)).encode(), digest_size=8).digest()
        margin += (int.from_bytes(h, "little") / 2**64 - 0.5) * 0.30
        if abs(margin) < 0.08:
            return "TIE"
        return "A" if margin > 0 else "B"


class HTTPJudge(Judge):
    def __init__(self, spec: str):
        import httpx
        # "anthropic" uses the env default; "anthropic:claude-sonnet-5" pins a model,
        # so two models from one provider can be run as two distinct judges.
        provider, _, model_override = spec.partition(":")
        self.provider = provider
        self.name = spec
        self.http = httpx.Client(timeout=45)
        if provider == "openai":
            self.key = os.environ["OPENAI_API_KEY"]
            self.model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        elif provider == "anthropic":
            self.key = os.environ["ANTHROPIC_API_KEY"]
            # Sonnet 5 and Opus 5 refuse the SYNTHETIC corpus outright
            # (stop_reason=refusal, categories bio / cyber): its generated
            # vocabulary reads as pseudo-biochemistry. Both answer real arXiv
            # abstracts cleanly. Haiku 4.5 answers either, so it is the default.
            self.model = os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5")
        else:
            raise SystemExit(f"unknown judge {provider}")
        if model_override:
            self.model = model_override

    RETRY_STATUS = (429, 500, 502, 503, 504)

    def _post(self, url: str, headers: dict, payload: dict, tries: int = 6):
        """Retry throttling and transient 5xx, honouring Retry-After.

        Without this a 10 RPM account fails most calls, and judge_duel turns every
        failure into a swap-inconsistent verdict -- so the harness reports its own
        throttling as evidence that the judge is position-biased.
        """
        last_exc = None
        for i in range(tries):
            try:
                r = self.http.post(url, headers=headers, json=payload)
            except Exception as e:
                # A transport failure is exactly the retryable case, and it was the
                # one case not retried: a stale pooled connection turned every
                # subsequent call into a 120s hang and killed a whole run.
                last_exc = e
                wait = min(2.0 ** i, 8.0)
                log(f"    judge {self.name} transport {type(e).__name__}, "
                    f"retry {i + 1}/{tries - 1} in {wait:.0f}s")
                self.http.close()
                import httpx as _hx
                self.http = _hx.Client(timeout=45)   # fresh pool, not the wedged one
                time.sleep(wait)
                continue
            # A 429 is only worth retrying if it is a RATE limit. Credit exhaustion
            # and a spent daily cap are not transient -- the body says "try again in
            # 8.64s" while the header says the window resets in 25 hours. Believe the
            # header. Retrying these is how a run spends half an hour going nowhere.
            spent = r.status_code == 429 and (
                "insufficient_quota" in r.text
                or "per day" in r.text
                or r.headers.get("x-ratelimit-remaining-requests") == "0")
            if r.status_code not in self.RETRY_STATUS or spent or i == tries - 1:
                r.raise_for_status()      # being out of credits is not a rate limit
                return r
            wait = float(r.headers.get("retry-after") or 0) or min(2.0 ** i, 8.0)
            log(f"    judge {self.name} http {r.status_code}, "
                f"retry {i + 1}/{tries - 1} in {wait:.0f}s")
            time.sleep(wait)
        if last_exc is not None:
            raise last_exc

    def compare(self, query, a, b):
        prompt = JUDGE_PROMPT.format(
            query=query,
            a="\n".join(f"{i+1}. {t[1]}" for i, t in enumerate(a)),
            b="\n".join(f"{i+1}. {t[1]}" for i, t in enumerate(b)))
        t = self.tracer
        gen_cm = (t.span("judge-verdict", as_type="generation", model=self.model,
                         input=[{"role": "user", "content": prompt}])
                  if t is not None else nullcontext(None))
        with gen_cm as gen:
          try:
            if self.provider == "openai":
                r = self._post(
                    "https://api.openai.com/v1/chat/completions",
                    {"Authorization": f"Bearer {self.key}"},
                    # Same instruction and the same room to answer as the Anthropic
                    # branch. Cross-judge agreement only means something if both
                    # judges were asked the same question; a terseness prompt on one
                    # side and a 4-token ceiling on the other is two questions.
                    {"model": self.model, "temperature": 0, "max_tokens": 256,
                     "messages": [
                         {"role": "system",
                          "content": "Reply with exactly one word: A, B, or TIE. "
                                     "No explanation, no analysis, no preamble."},
                         {"role": "user", "content": prompt}]})
                d = r.json()
                ch = (d.get("choices") or [{}])[0]
                # A reply cut off mid-sentence is not a verdict. Parsing it yields
                # TIE, because it starts with neither A nor B, and a fabricated tie
                # is invisible: it inflates the tie rate and the swap-inconsistency
                # rate while n_errors still reads zero.
                if ch.get("finish_reason") == "length":
                    raise RuntimeError("verdict truncated at max_tokens, not a TIE")
                out = (ch.get("message") or {}).get("content")
                if out is None:
                    raise RuntimeError(
                        f"no content (finish_reason={ch.get('finish_reason')})")
                u = d.get("usage") or {}
                usage = {"input": u.get("prompt_tokens", 0),
                         "output": u.get("completion_tokens", 0)}
            else:
                r = self._post(
                    "https://api.anthropic.com/v1/messages",
                    {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                    # `temperature` is removed on Sonnet 5+ (400), and thinking is
                    # ADAPTIVE by default, which would put a thinking block at
                    # content[0] and make the two order-swapped calls
                    # non-deterministic -- noise the swap test would read as bias.
                    {"model": self.model, "max_tokens": 256,
                     "thinking": {"type": "disabled"},
                     "system": "Reply with exactly one word: A, B, or TIE. "
                               "No explanation, no analysis, no preamble.",
                     "messages": [{"role": "user", "content": prompt}]})
                d = r.json()
                # A refusal is HTTP 200 with an EMPTY content array. Defaulting that
                # to "" makes every refusal parse as a considered TIE -- 160 refusals
                # once reported themselves as 80 clean ties and kappa exactly 0.000.
                if d.get("stop_reason") == "refusal":
                    cat = (d.get("stop_details") or {}).get("category")
                    raise RuntimeError(f"model refused (category={cat})")
                # A judge that spent its whole budget analysing did not answer the
                # question. Parsing that ramble gives TIE (it starts with neither A
                # nor B), which would be a fabricated verdict, so it is an error.
                if d.get("stop_reason") == "max_tokens":
                    raise RuntimeError("verdict truncated at max_tokens, not a TIE")
                out = next((blk["text"] for blk in d["content"]
                            if blk["type"] == "text"), None)
                if out is None:
                    raise RuntimeError(
                        f"no text block (stop_reason={d.get('stop_reason')})")
                u = d.get("usage") or {}
                usage = {"input": u.get("input_tokens", 0),
                         "output": u.get("output_tokens", 0)}
          except Exception as e:                     # a judge that errors is a TIE,
            body = getattr(getattr(e, "response", None), "text", "")
            log(f"    judge {self.name} error: {e}"  # never a silent win for either side
                + (f"\n      body: {body[:300]}" if body else ""))
            Tracer.set(gen, level="ERROR", status_message=str(e)[:400])
            return None
          norm = out.strip().upper()
          verdict = ("A" if norm.startswith("A")
                     else "B" if norm.startswith("B") else "TIE")
          Tracer.set(gen, output=out, usage_details=usage,
                     metadata={"verdict": verdict, "raw": out[:200]})
          return verdict


def judge_duel(judge: Judge, q: Query, a_ids: list[str], b_ids: list[str],
               text_of: Callable[[str], str], k: int = 5,
               tracer: Optional["Tracer"] = None) -> dict:
    """One paired verdict, run in both orders. Returns the de-biased verdict plus the
    evidence of position bias for this item."""
    A: list[Item] = [(c, text_of(c)[:220]) for c in a_ids[:k]]
    B: list[Item] = [(c, text_of(c)[:220]) for c in b_ids[:k]]
    cm = (tracer.span("judge-duel", as_type="evaluator",
                      input={"query": q.text,
                             "list_a": [t[1] for t in A],
                             "list_b": [t[1] for t in B]},
                      metadata={"query_id": q.id, "judge": judge.name,
                                "provenance": q.provenance})
          if tracer is not None else nullcontext(None))
    with cm as ev:
        if isinstance(judge, MockJudge):
            judge.bind(q.id)
        v1 = judge.compare(q.text, A, B)        # A first
        v2 = judge.compare(q.text, B, A)        # B first
        if v1 is None or v2 is None:
            Tracer.set(ev, output={"verdict": None, "reason": "judge call failed"},
                       level="WARNING")
            return dict(verdict=None, consistent=False, v1=v1, v2=v2)
        flip = {"A": "B", "B": "A", "TIE": "TIE"}
        v2n = flip[v2]                           # normalise back to A/B naming
        if v1 == v2n:
            verdict, consistent = v1, True
        else:
            verdict, consistent = "TIE", False   # disagreement under swap is not a win
        Tracer.set(ev, output={"verdict": verdict, "consistent_under_swap": consistent,
                               "a_first": v1, "b_first": v2n})
        return dict(verdict=verdict, consistent=consistent, v1=v1, v2=v2n)


# ----------------------------------------------------------------------------------
# §9  Langfuse tracing (optional, degrades to a local JSON log)
# ----------------------------------------------------------------------------------

class Tracer:
    """Langfuse tracing, or a clean and *loud* no-op.

    Targets the Langfuse v4 Python SDK. Older SDKs are refused with a message rather
    than detected-and-guessed: the previous shim probed for a v2 method and a v3
    method, found neither on v4, fell through to a call that does not exist, and
    swallowed the AttributeError on every single event while printing "connected".
    A tracer that reports itself healthy while emitting nothing is worse than none,
    so credentials are checked once at startup and the first failure is logged.

    Every number is still written to .proofline/reports/*.json, so the evidence
    exists whether or not tracing worked.
    """

    def __init__(self, cfg: Config):
        self.enabled = False
        self.client = None
        self.events: list[dict] = []
        self.session_id: Optional[str] = None
        self._obs = None
        self._warned = False
        if not cfg.langfuse:
            return
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_SECRET_KEY")
        if not (pk and sk):
            log("langfuse: no keys, tracing disabled (results still written to disk)")
            return
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        try:
            from langfuse import Langfuse
            client = Langfuse(public_key=pk, secret_key=sk, host=host)
        except Exception as e:
            log(f"langfuse: disabled ({type(e).__name__}: {e})")
            return
        obs = getattr(client, "start_as_current_observation", None)
        if obs is None:
            log("langfuse: installed SDK is older than v4 (no "
                "start_as_current_observation); tracing disabled")
            return
        # One round trip, and it converts the two failures that cost the most time
        # into a message: a secret key pasted into the public slot, and an EU host
        # for a US-region project. Both present as a bare 401 on every later call.
        try:
            ok = client.auth_check()
        except Exception as e:
            log(f"langfuse: {host} rejected the credentials ({type(e).__name__}). "
                "Check LANGFUSE_PUBLIC_KEY is the pk-lf- key, and that LANGFUSE_HOST "
                "matches your project's region (us. vs eu.). Tracing disabled.")
            return
        if not ok:
            log(f"langfuse: auth_check false at {host}; tracing disabled")
            return
        self.client, self._obs, self.enabled = client, obs, True
        log(f"langfuse: connected to {host}")

    def _warn(self, e: Exception) -> None:
        if not self._warned:
            self._warned = True
            log(f"langfuse: a tracing call failed ({type(e).__name__}: {e}); "
                "further tracing failures will not be reported")

    # -- observations ---------------------------------------------------------------
    def span(self, name: str, *, as_type: str = "span", **kw):
        """One observation, nested automatically under whatever span is active.

        Returns a no-op context manager when tracing is off, so call sites read the
        same either way. `as_type` must be a v4 observation type -- 'retriever' for a
        lookup, 'generation' for a model call, 'evaluator' for a judge -- because the
        type is what drives cost tracking and the agent graph.
        """
        if self.enabled:
            try:
                return self._obs(as_type=as_type, name=name, **kw)
            except Exception as e:
                self._warn(e)
        return nullcontext(None)

    def session(self, session_id: str):
        """Group this run's four branch traces into one session."""
        self.session_id = session_id
        if self.enabled:
            try:
                from langfuse import propagate_attributes
                return propagate_attributes(session_id=session_id)
            except Exception as e:
                self._warn(e)
        return nullcontext(None)

    @staticmethod
    def set(obs, **kw) -> None:
        """Update an observation that may be a no-op."""
        if obs is not None:
            try:
                obs.update(**kw)
            except Exception:
                pass

    def event(self, name: str, **payload) -> None:
        self.events.append(dict(name=name, ts=time.time(), **payload))
        if not self.enabled:
            return
        try:
            self.client.create_event(name=name, input=payload.get("input"),
                                     output=payload.get("output"))
        except Exception as e:
            self._warn(e)

    def score(self, name: str, value: float, *, comment: str = "") -> None:
        self.events.append(dict(name="score", metric=name, value=value, comment=comment))
        if not self.enabled:
            return
        try:
            self.client.score_current_trace(name=name, value=float(value),
                                            data_type="NUMERIC",
                                            comment=comment or None)
        except Exception as e:
            self._warn(e)

    def flush(self) -> None:
        if self.enabled:
            try:
                self.client.flush()
            except Exception as e:
                self._warn(e)


# ----------------------------------------------------------------------------------
# §10  Harness: run scorers, pool, score
# ----------------------------------------------------------------------------------

@dataclass
class RunResult:
    scorer: str
    label_src: str
    corpus_n: int
    ranked: dict[str, list[str]]              # query_id -> ranked card ids
    top_score: dict[str, float]
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    timing_ms: float = 0.0


def load_queries(store: Store, provenance: str, include_dropped: bool = False) -> list[Query]:
    with store.conn.cursor() as cur:
        cur.execute("SELECT * FROM queries WHERE provenance = %s AND (dropped IS NULL OR %s) "
                    "ORDER BY id", (provenance, include_dropped))
        qs = cur.fetchall()
        cur.execute("SELECT q.id qid, r.card_id, r.grade, r.slot FROM queries q "
                    "JOIN qrels r ON r.query_id = q.id WHERE q.provenance = %s",
                    (provenance,))
        rel: dict[str, dict[str, float]] = defaultdict(dict)
        slots: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
        for r in cur.fetchall():
            rel[r["qid"]][r["card_id"]] = float(r["grade"])
            if r["slot"] is not None:
                slots[r["qid"]][r["slot"]].add(r["card_id"])
    return [Query(id=q["id"], text=q["text"], provenance=q["provenance"],
                  as_of=q["as_of"], source_card=q["source_card"],
                  rel=rel.get(q["id"], {}), slices=list(q["slices"]),
                  answerable=q["answerable"], intent=q["intent"], family=q["family"],
                  slots=[s for _i, s in sorted(slots[q["id"]].items())],
                  answer=q["answer"], pair_id=q["pair_id"]) for q in qs]


_INDEX_CACHE: dict[Any, Index] = {}


def build_index(store: Store, n_limit: Optional[int],
                keep: Optional[frozenset] = None) -> Index:
    """One index over the whole record; snapshots are applied as a mask at query time.

    `n_limit` sets the corpus size for a B3 scale point; `keep` names the cards that
    must survive the sampling (every judged document), so growing the corpus adds
    distractors rather than removing answers.
    """
    key = ("idx", n_limit, keep)
    hit = _INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    rows = store.subset(keep, n_limit)
    idx = Index(rows)
    with store.conn.cursor() as cur:
        cur.execute("SELECT src, dst, valid_from FROM edges WHERE kind IN ('parent','citation')")
        edges = [(r["src"], r["dst"], r["valid_from"]) for r in cur.fetchall()]
    idx.attach_graph(edges)
    _INDEX_CACHE[key] = idx
    return idx


def execute_run(store: Store, scorer: Scorer, embedder: Embedder,
                queries: Sequence[Query], label_src: str, corpus_n: Optional[int],
                k: int, tracer: Tracer,
                threshold: Optional[float] = None,
                keep: Optional[frozenset] = None) -> RunResult:
    """One scorer over one query set.

    The index covers the whole record; each query is restricted to the corpus as it
    stood at its own `as_of` by a mask inside the scorer. A frozen query set scored
    against today's corpus quietly penalises whichever scorer finds the current version
    of a card, so the snapshot is not optional and cannot be a later optimisation."""
    with tracer.span("retrieve-candidates", as_type="retriever",
                     input=dict(scorer=scorer.name, label_src=label_src,
                                queries=len(queries)),
                     metadata=dict(corpus_n=corpus_n, topk=k,
                                   abstain_threshold=threshold)) as obs:
        t0 = time.perf_counter()
        ranked: dict[str, list[str]] = {}
        tops: dict[str, float] = {}
        if queries:
            idx = build_index(store, corpus_n, keep)
            scorer.prepare(idx, embedder)
            for q in queries:
                out = scorer.run(q, k)
                tops[q.id] = out.top_score
                # Abstention is part of retrieval, not a post-hoc filter on the report.
                # A scorer that cannot return nothing will answer every no-answer query,
                # and the hard check below would then be unfalsifiable.
                if threshold is not None and out.top_score < threshold:
                    ranked[q.id] = []
                else:
                    ranked[q.id] = out.ids
        ms = (time.perf_counter() - t0) * 1000
        Tracer.set(obs, output=dict(queries=len(queries), ms=round(ms, 1),
                                    abstained=sum(1 for v in ranked.values() if not v)))
        return RunResult(scorer.name, label_src, corpus_n or 0, ranked, tops,
                         timing_ms=ms)


def calibrate_abstain(store: Store, scorer: Scorer, embedder: Embedder,
                      dev: Sequence[Query], cfg: Config, tracer: Tracer) -> float:
    """Pick the threshold that makes this scorer abstain on `abstain_fabr` of queries
    that DO have an answer. Two scorers compared at their own natural thresholds are
    being compared on their thresholds; holding the false-abstention rate equal is what
    makes the false-answer rates comparable at all.

    Calibrated on the dev split only. A threshold fitted on the queries you then report
    is a threshold fitted to the report."""
    if not dev:
        return float("-inf")
    r = execute_run(store, scorer, embedder, dev, "calibration", None, 1, tracer)
    vals = sorted(r.top_score.values())
    if not vals:
        return float("-inf")
    return float(np.quantile(vals, cfg.abstain_fabr))


def score_runs(runs: list[RunResult], queries: Sequence[Query], k: int) -> None:
    """Second pass: build the judgment pool across all runs, then score. Pooling has to
    happen after every run exists, which is why metrics are not computed inline."""
    qmap = {q.id: q for q in queries}
    pool: dict[str, set[str]] = defaultdict(set)
    for r in runs:
        for qid, ids in r.ranked.items():
            pool[qid].update(ids[:k])
    for r in runs:
        for qid, ids in r.ranked.items():
            q = qmap[qid]
            if not q.answerable:
                continue
            judged = set(q.rel) | pool[qid]
            r.per_query[qid] = dict(
                ndcg=ndcg_at_k(ids, q.rel, k, condensed=True, judged=judged),
                ndcg_raw=ndcg_at_k(ids, q.rel, k, condensed=False),
                mrr=mrr(ids, q.rel),
                recall=recall_at_k(ids, q.rel, k),
                bpref=bpref(ids, q.rel, pool[qid]),
            )


def paired(a: RunResult, b: RunResult, metric: str,
           qids: Optional[Iterable[str]] = None) -> tuple[list[float], list[str]]:
    ids = list(qids) if qids is not None else sorted(set(a.per_query) & set(b.per_query))
    d, keep = [], []
    for q in ids:
        if q in a.per_query and q in b.per_query:
            x, y = a.per_query[q].get(metric), b.per_query[q].get(metric)
            if x is None or y is None or math.isnan(x) or math.isnan(y):
                continue
            d.append(y - x)
            keep.append(q)
    return d, keep


# ---- hard checks: one occurrence blocks. Never averaged, never traded off. ---------

def hard_checks(run: RunResult, idx_rows: dict, queries: Sequence[Query],
                k: int) -> dict[str, dict]:
    qmap = {q.id: q for q in queries}
    superseded_above = []
    out_of_snapshot = []
    answered_no_answer = []
    succ_of = {}
    for cid, r in idx_rows.items():
        if r["supersedes_id"]:
            succ_of[r["supersedes_id"]] = cid
    for qid, ids in run.ranked.items():
        q = qmap.get(qid)
        if q is None:
            continue
        top = ids[:k]
        seen_pos = {c: i for i, c in enumerate(top)}
        for c in top:
            s = succ_of.get(c)
            if s is not None and s in seen_pos and seen_pos[s] > seen_pos[c]:
                superseded_above.append((qid, c, s))
            row = idx_rows.get(c)
            if row is not None and row["committed_at"] > q.as_of:
                out_of_snapshot.append((qid, c))
        if not q.answerable and top:
            answered_no_answer.append(qid)
    return {
        "superseded_above_successor": dict(
            fails=len(superseded_above), of=len(run.ranked),
            examples=superseded_above[:3]),
        "result_outside_snapshot": dict(
            fails=len(out_of_snapshot), of=len(run.ranked),
            examples=out_of_snapshot[:3]),
        "answered_a_no_answer_query": dict(
            fails=len(answered_no_answer),
            of=sum(1 for q in queries if not q.answerable),
            examples=answered_no_answer[:3]),
    }


def guarded_span(src: str) -> str:
    """The retrieval/scoring path: from the §4 banner LINE to the §11 banner LINE.

    The split strings start with a newline, so they match the banner lines and never
    these literals, whose source text carries a backslash where the newline would be.
    A renamed §4 banner stops the run; a renamed §11 banner stretches the span into
    B1's planted reads, which trip the guard. A broken banner always fails loudly."""
    _head, found, rest = src.partition("\n# §4  Retrieval index")
    if not found:
        raise SystemExit("truth guard: the '# §4  Retrieval index' banner is missing, "
                         "so there is no span to check")
    return rest.split("\n# §11  Branch")[0]


def truth_leak_in(src: str) -> Optional[str]:
    """The first planted-truth literal on the retrieval/scoring path, or None. Every
    planted table is named with the true_ prefix, and none of them may be read there."""
    m = re.search(r"\btrue_(?:support|mention|fact|entit)\w*", guarded_span(src))
    return m.group(0) if m else None


def assert_no_truth_leak(store: Store) -> None:
    """The planted truth lives in the same database as everything else, which is a
    hazard. Nothing on the retrieval path may read it. This is cheap and it is the
    kind of guard that stops a harness quietly grading itself."""
    # utf-8 explicitly: the banners carry a '§', and Windows' default codec decodes it
    # into something the split in truth_leak_in never finds.
    hit = truth_leak_in(Path(__file__).read_text(encoding="utf-8"))
    if hit:
        raise SystemExit(f"TRUTH LEAK: the retrieval/scoring path references {hit}")

# ----------------------------------------------------------------------------------
# §11  Branch experiments
# ----------------------------------------------------------------------------------

def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:+.{nd}f}" if abs(x) < 1 and nd >= 3 else f"{x:.{nd}f}"


def verdict_of(mean: float, lo: float, hi: float, tol: float = 0.0) -> str:
    if math.isnan(lo):
        return "NO DATA"
    if lo > tol:
        return "BETTER"
    if hi < (-abs(tol) if tol else 0.0):
        return "WORSE"
    return "UNDERPOWERED"


def build_usage_mined(store: Store, cfg: Config, embedder: Embedder,
                      tracer: Tracer) -> int:
    """Simulate a click log. The 'user' only ever saw the incumbent's top 3, so the
    label is whatever the incumbent surfaced. This set rewards the incumbent by
    construction; B1 measures how badly."""
    planted = [q for q in load_queries(store, cfg.eval_provenance) if q.answerable]
    inc = SCORERS[cfg.baseline]()
    run = execute_run(store, inc, embedder, planted, "usage_probe", None, 3, tracer)
    rows_q, rows_r = [], []
    for i, q in enumerate(planted):
        top3 = run.ranked.get(q.id, [])
        clicked = next((c for c in top3 if q.rel.get(c, 0) > 0), top3[0] if top3 else None)
        if clicked is None:
            continue
        # A fresh id space. Deriving it from the source id (the old
        # q.id.replace("q_pl_", "q_us_")) silently produced the SOURCE id whenever the
        # evaluated provenance was not "planted", colliding on the primary key.
        qid = f"q_us_{i:06d}"
        rows_q.append((qid, "usage_mined", q.text, q.as_of, q.source_card, True, q.slices))
        rows_r.append((qid, clicked, 2.0))
    with store.conn.cursor() as cur:
        cur.execute("DELETE FROM qrels WHERE query_id IN "
                    "(SELECT id FROM queries WHERE provenance='usage_mined')")
        cur.execute("DELETE FROM queries WHERE provenance='usage_mined'")
        with cur.copy("COPY queries (id,provenance,text,as_of,source_card,answerable,"
                      "slices) FROM STDIN") as cp:
            for r in rows_q:
                cp.write_row(r)
        with cur.copy("COPY qrels (query_id,card_id,grade) FROM STDIN") as cp:
            for r in rows_r:
                cp.write_row(r)
    return len(rows_q)


def filter_disconnected(store: Store, cfg: Config, embedder: Embedder) -> dict:
    """MuSiQue's disconnection test. For every query with a last hop, hide the first
    hop's card and ask single-hop BM25 and dense for the answer's card. If either finds
    it in the top k, the question never needed the hop: drop it, and drop its pair with
    it, because half a contrast pair measures nothing. Re-running resets and recomputes."""
    counts: dict = {f: {"kept": 0, "dropped": 0} for f in LAST_HOP_FAMILIES}
    counts["pairs_dropped"] = 0
    every = load_queries(store, "multihop", include_dropped=True)
    qs = [q for q in every if q.answerable and q.family in LAST_HOP_FAMILIES]
    with store.conn.cursor() as cur:
        cur.execute("UPDATE queries SET dropped = NULL WHERE provenance = 'multihop'")
    if not qs:
        return counts
    idx = build_index(store, None)
    single: set[str] = set()
    for q in qs:
        m = idx.snapshot_mask(q.as_of).copy()
        for cid in q.slots[0]:
            if cid in idx.pos:
                m[idx.pos[cid]] = False
        lex = np.where(m, idx.bm25(q.text), -np.inf)
        den = np.where(m, idx.cosine(embedder.encode_queries([q.text])[0]), -np.inf)
        top = {idx.ids[i] for i in topn(lex, cfg.topk) if np.isfinite(lex[i]) and lex[i] > 0}
        top |= {idx.ids[i] for i in topn(den, cfg.topk) if np.isfinite(den[i])}
        hit = bool(top & q.slots[-1])
        counts[q.family]["dropped" if hit else "kept"] += 1
        if hit:
            single.add(q.id)
    bad_groups = {q.pair_id for q in every if q.id in single and q.pair_id}
    paired = {q.id for q in every if q.pair_id in bad_groups and q.id not in single}
    counts["pairs_dropped"] = len(paired)
    for q in every:          # a kept half of a dropped pair leaves the kept count
        if q.id in paired and q.family in LAST_HOP_FAMILIES:
            counts[q.family]["kept"] -= 1
            counts[q.family]["dropped"] += 1
    with store.conn.cursor() as cur:
        cur.execute("UPDATE queries SET dropped = 'single_hop' WHERE id = ANY(%s)", (sorted(single),))
        cur.execute("UPDATE queries SET dropped = 'pair_dropped' WHERE id = ANY(%s)", (sorted(paired),))
    return counts


def _norm_surface(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _norm_class(s: Optional[str]) -> str:
    return re.sub(r"[^a-z]+", "", (s or "").lower())


def _norm_value(v: Optional[str]) -> Optional[str]:
    m = re.search(r"\d*\.\d+|\d+", str(v)) if v is not None else None
    return f"{float(m.group(0)):.2f}" if m else None


def _snake(pred: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pred.strip()).lower().replace(" ", "_")


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _strongest(a: str, b: str) -> str:
    return min(a, b, key=lambda r: MENTION_ROLES.index(r) if r in MENTION_ROLES else 99)


def extraction_fidelity(truth: dict, ext: dict) -> dict:
    """Grade one extractor against the planted entity layer, on the cards it actually read.

    Extracted entities carry their own ids, so they are aligned to planted ones through
    surface forms. A mention whose surface nothing planted (the LLM turning "a sampling
    method" into an entity) is a false positive; an extracted entity whose mentions align
    to two planted ones is a false merge; a planted entity whose aliases landed on two
    extracted ids is an unresolved alias."""
    cards = ext["cards"]
    t_ent = truth["entities"]
    t_pairs: dict[tuple[str, str], str] = {}
    for c, e, _s, r in truth["mentions"]:
        if c in cards:
            t_pairs[(c, e)] = _strongest(t_pairs.get((c, e), r), r)
    t_fac = {(c, s, r, o, _norm_value(v) if r == "reports" else None)
             for c, s, r, o, v in truth["facts"] if c in cards}
    surf2t: dict[str, str] = {}
    for eid, (name, _k, _p) in sorted(t_ent.items()):
        surf2t.setdefault(_norm_surface(name), eid)
    for _c, eid, surface, _r in sorted(truth["mentions"]):
        surf2t.setdefault(_norm_surface(surface), eid)

    x_pairs: dict[tuple[str, str], str] = {}
    votes: dict[str, Counter] = defaultdict(Counter)
    surf_by_t: dict[str, set] = defaultdict(set)
    xs_by_t: dict[str, set] = defaultdict(set)
    for c, xid, surface, role in sorted(ext["mentions"]):
        ns = _norm_surface(surface)
        tid = surf2t.get(ns)
        key = (c, tid) if tid else (c, f"?{xid}:{ns}")
        x_pairs[key] = _strongest(x_pairs.get(key, role), role)
        if tid:
            votes[xid][tid] += 1
            surf_by_t[tid].add(ns)
            xs_by_t[tid].add(xid)
    hit = set(t_pairs) & set(x_pairs)
    x2t = {x: v.most_common(1)[0][0] for x, v in votes.items()}

    t2x: dict[str, Counter] = defaultdict(Counter)
    for x, v in votes.items():
        for t, n in v.items():
            t2x[t][x] += n
    type_scores = []
    for tid, xs in t2x.items():
        if tid in t_ent:
            _n, leaf, parent = t_ent[tid]
            kind = _norm_class(ext["entities"].get(xs.most_common(1)[0][0], ("", ""))[1])
            type_scores.append(1.0 if kind == _norm_class(leaf)
                               else 0.5 if parent and kind == _norm_class(parent) else 0.0)
    eligible = [t for t, ss in surf_by_t.items() if len(ss) > 1]

    def onto_class(k: str) -> Optional[str]:
        return next((c for c in ONTO_CLASSES if _norm_class(c) == _norm_class(k)), None)

    kind_of = {x: onto_class(k) for x, (_n, k) in ext["entities"].items()}
    x_fac: set = set()
    untyped = violations = 0
    for c, s, r, o, v in ext["facts"]:
        rel = _snake(r)
        if rel not in ONTO_RELATIONS:
            untyped += 1
            continue
        x_fac.add((c, x2t.get(s, f"?{s}"), rel, x2t.get(o, f"?{o}"),
                   _norm_value(v) if rel == "reports" else None))
        dom, rng_ = ONTO_RELATIONS[rel]
        ks, ko = kind_of.get(s), kind_of.get(o)
        if not ks or not ko or top_class(ks) not in dom or top_class(ko) not in rng_:
            violations += 1
    f_hit = t_fac & x_fac

    x_classes = {_norm_class(c) for c, _p in ext["classes"]}
    x_edges = {(_norm_class(c), _norm_class(p)) for c, p in ext["classes"] if p}
    planted_edges = {(_norm_class(c), _norm_class(p)) for c, p in ONTO_CLASSES.items() if p}

    methods = {e for e, (_n, k, _p) in t_ent.items() if top_class(k) == "Method"}
    users: dict[str, set] = defaultdict(set)
    definers: dict[str, set] = defaultdict(set)
    for (c, e), r in t_pairs.items():
        if e in methods and r == "uses":
            users[e].add(c)
        if e in methods and r == "introduces":
            definers[e].add(c)
    for c, s, r, _o, _v in t_fac:
        if r == "reports" and s in methods:
            definers[s].add(c)
    pairs = [(u, d, e) for e in methods for u in users[e] for d in definers[e] if u != d]
    covered = sum(1 for u, d, e in pairs if (u, e) in hit and (d, e) in hit)
    return dict(
        mention_precision=_ratio(len(hit), len(x_pairs)),
        mention_recall=_ratio(len(hit), len(t_pairs)),
        role_accuracy=_ratio(sum(1 for p in hit if x_pairs[p] == t_pairs[p]), len(hit)),
        type_accuracy=statistics.fmean(type_scores) if type_scores else float("nan"),
        alias_resolution=_ratio(sum(1 for t in eligible if len(xs_by_t[t]) == 1), len(eligible)),
        false_merges=sum(1 for v in votes.values() if len(v) > 1),
        fact_precision=_ratio(len(f_hit), len(x_fac)), fact_recall=_ratio(len(f_hit), len(t_fac)),
        untyped_links=untyped, domain_range_violations=violations,
        class_recall=len({_norm_class(c) for c in ONTO_CLASSES} & x_classes) / len(ONTO_CLASSES),
        subclass_edge_recall=len(planted_edges & x_edges) / len(planted_edges),
        bridge_coverage=_ratio(covered, len(pairs)), bridges=len(pairs))


def grade_source(truth: dict, ext: dict) -> dict:
    counts = dict(cards=len(ext["cards"]), entities=len(ext["entities"]),
                  mentions=len(ext["mentions"]), facts=len(ext["facts"]))
    if not truth["entities"]:
        return dict(measured=False, **counts)
    return dict(measured=True, **counts, **extraction_fidelity(truth, ext))


_FIDELITY_ROWS = [
    ("mention_precision", "mention precision"), ("mention_recall", "mention recall"),
    ("type_accuracy", "type accuracy (parent class scores 0.5)"),
    ("alias_resolution", "alias resolution"), ("false_merges", "false merges (count)"),
    ("fact_precision", "typed fact precision"), ("fact_recall", "typed fact recall"),
    ("untyped_links", "untyped links (count)"),
    ("domain_range_violations", "domain/range violations (count)"),
    ("class_recall", "ontology class recall"), ("subclass_edge_recall", "subclass edge recall"),
    ("bridges", "bridges graded (count)"),
    ("bridge_coverage", "bridge coverage (both ends extracted)"),
    ("role_accuracy", "role accuracy (cue phrases: template-tuned)"),
]


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def extraction_block(store: Store) -> Optional[dict]:
    """B1's grade of every extractor that has written to this corpus. Printed and returned."""
    with store.conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM extracted_cards ORDER BY source")
        sources = [r["source"] for r in cur.fetchall()]
        if not sources:
            log("\n  extraction fidelity: NOT INGESTED. Run `uv run ingest_semantica.py "
                "--method regex` (or --method llm) after seeding; `all` re-seeds and empties it.")
            return None
        cur.execute("SELECT id, name, kind, parent_kind FROM true_entities")
        truth = dict(entities={r["id"]: (r["name"], r["kind"], r["parent_kind"]) for r in cur.fetchall()})
        cur.execute("SELECT card_id, entity_id, surface, role FROM true_mentions")
        truth["mentions"] = [tuple(r.values()) for r in cur.fetchall()]
        cur.execute("SELECT card_id, subj, rel, obj, value FROM true_facts")
        truth["facts"] = [tuple(r.values()) for r in cur.fetchall()]
        out: dict[str, dict] = {}
        for src in sources:
            cur.execute("SELECT card_id FROM extracted_cards WHERE source = %s", (src,))
            ext = dict(cards={r["card_id"] for r in cur.fetchall()})
            cur.execute("SELECT id, name, kind FROM entities WHERE source = %s", (src,))
            ext["entities"] = {r["id"]: (r["name"], r["kind"]) for r in cur.fetchall()}
            cur.execute("SELECT card_id, entity_id, surface, role FROM mentions WHERE source = %s", (src,))
            ext["mentions"] = [tuple(r.values()) for r in cur.fetchall()]
            cur.execute("SELECT card_id, subj, rel, obj, value FROM facts WHERE source = %s", (src,))
            ext["facts"] = [tuple(r.values()) for r in cur.fetchall()]
            cur.execute("SELECT class, parent FROM onto_classes WHERE source = %s", (src,))
            ext["classes"] = [(r["class"], r["parent"]) for r in cur.fetchall()]
            out[src] = grade_source(truth, ext)
    log("\n  extraction fidelity against the planted entity layer (on the cards each extractor read):")
    log("    " + f"{'':<46}" + "".join(f"{s[-26:]:>28}" for s in sources))
    rows = [("cards", "cards read"), ("entities", "entities"), ("mentions", "mentions"),
            ("facts", "facts")]
    if any(out[s]["measured"] for s in sources):
        rows += _FIDELITY_ROWS
    else:
        log("    NOT MEASURABLE: this corpus has no planted entities (real data); counts only.")
    for key, label in rows:
        log("    " + f"{label:<46}" + "".join(f"{_cell(out[s].get(key)):>28}" for s in sources))
    return out


def branch1_labels(store: Store, cfg: Config, embedder: Embedder,
                   tracer: Tracer) -> dict:
    """B1: Gold from nowhere.

    The question the proofline asks is 'where do labels come from'. The question that
    actually decides anything is 'does a label source pick the same winner the truth
    would have picked'. A label source can be badly calibrated and still be a perfectly
    good SELECTOR, and it can look precise and still choose wrong. So we measure the
    verdict, not the label.
    """
    rule("B1  GOLD FROM NOWHERE: do mined labels pick the same winner as the truth?")
    a_name, b_name = cfg.baseline, cfg.candidate
    out: dict[str, Any] = {"sources": {}}
    truth_verdict = None

    for src in ["planted", "dag_mined", "generated", "usage_mined"]:
        # On a real corpus there is no planted truth; that row simply disappears.
        qs = [q for q in load_queries(store, src) if q.answerable]
        if not qs:
            continue
        runs = []
        refused = None
        for sc_name in [a_name, b_name]:
            sc = SCORERS[sc_name]()
            r = check_circularity(sc, src)
            if r:
                refused = r
                break
            runs.append(execute_run(store, sc, embedder, qs, src, None, cfg.topk, tracer))
        if refused:
            log(f"  {src:<12} {refused}")
            out["sources"][src] = dict(refused=refused)
            continue
        score_runs(runs, qs, cfg.topk)
        d, _ = paired(runs[0], runs[1], "ndcg")
        mean, lo, hi = bootstrap_ci(d, cfg.bootstrap)
        pw = paired_power(d, cfg.target_effect); pw.pop("n", None)
        v = verdict_of(mean, lo, hi)
        if src == "planted":
            truth_verdict = v
        out["sources"][src] = dict(n=len(d), mean=mean, lo=lo, hi=hi, verdict=v, **pw)
        log(f"  {src:<12} n={len(d):<4} Δ nDCG@{cfg.topk}={fmt(mean)} "
            f"[{fmt(lo)},{fmt(hi)}]  MDE={fmt(pw['mde'])}  -> {v}")
        tracer.score(f"b1.{src}.delta_ndcg", mean, comment=v)

    # -- how faithful is the observable link structure to what actually grounded a card
    with store.conn.cursor() as cur:
        cur.execute("""
          WITH t AS (SELECT src,dst FROM edges WHERE kind='true_support'),
               p AS (SELECT src,dst FROM edges WHERE kind IN ('parent','citation'))
          SELECT (SELECT count(*) FROM t) AS n_true,
                 (SELECT count(*) FROM p) AS n_rec,
                 (SELECT count(*) FROM t JOIN p USING (src,dst)) AS n_both""")
        e = cur.fetchone()
    if not e["n_true"]:
        # No planted truth: there is nothing to compare the recorded links against.
        # Reporting 0.000/0.000 here (and a rank bias of "infx") states a measurement
        # that was never made, which is precisely the failure this harness is about.
        out["link_fidelity"] = None
        out["rank_bias"] = None
        log("\n  link fidelity: NOT MEASURABLE on this corpus -- it needs planted "
            "true_support,\n    and a real record has none. The recorded links are all "
            "there is.")
        log("  rank bias:     NOT MEASURABLE for the same reason.")
    else:
        lr = e["n_both"] / max(1, e["n_true"])
        lp = e["n_both"] / max(1, e["n_rec"])
        out["link_fidelity"] = dict(recall=lr, precision=lp,
                                    n_true=e["n_true"], n_recorded=e["n_rec"])
        log("\n  link fidelity of the record's own edges vs what actually grounded "
            "a card:")
        log(f"    recall {lr:.3f}  precision {lp:.3f}   "
            f"({e['n_both']} of {e['n_true']} true links were written down)")

    # -- rank bias: are the recorded links the ones the incumbent already surfaces?
    #    Needs planted truth to define an "unrecorded but true" link at all.
    if e["n_true"]:
        planted = [q for q in load_queries(store, cfg.eval_provenance) if q.answerable][:200]
        inc = SCORERS[cfg.baseline]()
        r = execute_run(store, inc, embedder, planted, "bias_probe", None, 3, tracer)
        hit_recorded = hit_unrecorded = tot_rec = tot_unrec = 0
        with store.conn.cursor() as cur:
            cur.execute("SELECT src,dst,kind FROM edges WHERE kind<>'citation'")
            rec: dict[str, set[str]] = defaultdict(set)
            tru: dict[str, set[str]] = defaultdict(set)
            for row in cur.fetchall():
                (rec if row["kind"] == "parent" else tru)[row["src"]].add(row["dst"])
        for q in planted:
            if not q.source_card:
                continue
            top3 = set(r.ranked.get(q.id, [])[:3])
            recorded = rec[q.source_card]
            unrecorded = tru[q.source_card] - recorded
            tot_rec += len(recorded); tot_unrec += len(unrecorded)
            hit_recorded += len(recorded & top3); hit_unrecorded += len(unrecorded & top3)
        br, bu = hit_recorded / max(1, tot_rec), hit_unrecorded / max(1, tot_unrec)
        out["rank_bias"] = dict(recorded_in_top3=br, unrecorded_in_top3=bu,
                                ratio=br / bu if bu else float("inf"))
        log(f"\n  rank bias (card 6SNVC4, measured rather than argued):")
        log(f"    a link the author DID record is in the incumbent's top-3 {br:.1%} of the time")
        log(f"    a link they did NOT record is in it {bu:.1%} of the time")
        log(f"    -> mined labels favour the incumbent by {br/bu if bu else float('inf'):.2f}x")

    out["extraction"] = extraction_block(store)
    log(f"\n  verdict agreement: truth says {truth_verdict}; " +
        ", ".join(f"{k}={v.get('verdict','-')}" for k, v in out["sources"].items()
                  if k != "planted" and "verdict" in v))
    out["truth_verdict"] = truth_verdict
    return out


def branch2_judge(store: Store, cfg: Config, embedder: Embedder,
                  tracer: Tracer) -> dict:
    """B2: Judge or metric. The judge is calibrated here, not trusted."""
    rule("B2  JUDGE OR METRIC: calibrating the instrument before believing it")
    qs = [q for q in load_queries(store, cfg.eval_provenance) if q.answerable][:cfg.judge_pairs]
    if not qs:
        return {}
    runs = []
    for nm in [cfg.baseline, cfg.candidate]:
        runs.append(execute_run(store, SCORERS[nm](), embedder, qs, "planted",
                                None, cfg.topk, tracer))
    score_runs(runs, qs, cfg.topk)

    with store.conn.cursor() as cur:
        cur.execute("SELECT id, title, one_line FROM cards")
        txt = {r["id"]: f"{r['title']}: {r['one_line']}" for r in cur.fetchall()}

    truth = {q.id: q.rel for q in qs}
    metric_verdict = {}
    for q in qs:
        a = runs[0].per_query.get(q.id, {}).get("ndcg", float("nan"))
        b = runs[1].per_query.get(q.id, {}).get("ndcg", float("nan"))
        if math.isnan(a) or math.isnan(b):
            continue
        metric_verdict[q.id] = "TIE" if abs(b - a) < 1e-9 else ("B" if b > a else "A")

    out: dict[str, Any] = {"judges": {}}
    verdicts_by_judge: dict[str, dict[str, Optional[str]]] = {}
    for jname in cfg.judges:
        judge: Judge = (MockJudge(truth) if jname == "mock"
                        else HTTPJudge(jname))
        judge.tracer = tracer
        if jname == "mock":
            log("  NOTE: 'mock' is a simulator with injected biases, not a model. "
                "Its numbers are evidence about the harness, not about a judge.")
        vs, inconsistent, first_pref, asked, identical, decided = {}, 0, 0, 0, 0, 0
        errors = 0
        t0 = time.perf_counter()
        for q in qs:
            a_ids = runs[0].ranked.get(q.id, [])
            b_ids = runs[1].ranked.get(q.id, [])
            if a_ids[:5] == b_ids[:5]:
                identical += 1          # nothing to prefer; asking anyway measures bias
                vs[q.id] = "TIE"
                continue
            d = judge_duel(judge, q, a_ids, b_ids, lambda c: txt.get(c, c),
                           tracer=tracer)
            asked += 1
            vs[q.id] = d["verdict"]
            if d["verdict"] is None:
                errors += 1                 # an API failure is not a swap flip
            elif not d["consistent"]:
                inconsistent += 1
            if d["v1"] in ("A", "B"):
                decided += 1
                if d["v1"] == "A":
                    first_pref += 1
        el = time.perf_counter() - t0
        verdicts_by_judge[jname] = vs
        common = [q.id for q in qs if q.id in metric_verdict and vs.get(q.id)]
        k_metric = cohen_kappa([metric_verdict[i] for i in common],
                               [vs[i] for i in common])
        pos_bias = inconsistent / max(1, asked - errors)
        first_rate = first_pref / max(1, decided)
        out["judges"][jname] = dict(n_asked=asked, n_identical=identical,
                                    n_errors=errors,
                                    kappa_vs_metric=k_metric,
                                    swap_inconsistency=pos_bias,
                                    prefers_first=first_rate, seconds=round(el, 1))
        log(f"  {jname:<12} asked={asked} (skipped {identical} where both scorers "
            f"returned the same top-5)"
            + (f"  [{errors} API ERRORS, not verdicts]" if errors else ""))
        log(f"  {'':<12} kappa vs nDCG={fmt(k_metric)}  flips under order "
            f"swap={pos_bias:.1%}  picks-first={first_rate:.1%}  ({el:.0f}s)")
        tracer.score(f"b2.{jname}.kappa_vs_metric", k_metric)
        tracer.score(f"b2.{jname}.swap_inconsistency", pos_bias)

    names = list(verdicts_by_judge)
    if len(names) > 1:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                common = [q.id for q in qs
                          if verdicts_by_judge[a].get(q.id) and verdicts_by_judge[b].get(q.id)]
                kk = cohen_kappa([verdicts_by_judge[a][x] for x in common],
                                 [verdicts_by_judge[b][x] for x in common])
                out.setdefault("judge_agreement", {})[f"{a}~{b}"] = kk
                log(f"  judge-vs-judge kappa {a} ~ {b}: {fmt(kk)}"
                    f"   {'(two judges that disagree cannot both be the yardstick)' if kk < 0.6 else ''}")

    log("\n  reading: a judge whose verdict flips when you swap the order is not "
        "measuring quality on those items.\n  Swap-inconsistency is the fraction of "
        "items where that happened; they are forced to TIE here rather than\n  counted "
        "as a win, which is the cheapest available de-biasing and still not a fix.")
    return out


def branch3_scale(store: Store, cfg: Config, embedder: Embedder,
                  tracer: Tracer) -> dict:
    """B3: The long record: scale, distractors, and abstention."""
    rule("B3  THE LONG RECORD: scale, distractor taxonomy, and saying nothing")
    qs = [q for q in load_queries(store, cfg.eval_provenance) if q.answerable]
    if not qs:
        log(f"  no answerable queries with provenance '{cfg.eval_provenance}'; "
            f"nothing to measure.\n  (--source arxiv defaults to dag_mined; "
            f"--eval-provenance overrides it.)")
        return {}
    # Every judged document is pinned into EVERY scale point, so the judged set has to
    # fit inside the smallest one. Real papers cite ~19 others, so sampling by query
    # COUNT silently pins more documents than the small point holds; Store.subset then
    # returns all the needles and the "6k" point is really 30k. The curve comes out
    # wrong and entirely plausible -- the Branch C trap one level up. Sample against a
    # document budget instead, and print the pinned count so it stays visible.
    if cfg.scale_judged_budget:
        rs = random.Random(cfg.seed)
        pool = list(qs); rs.shuffle(pool)
        picked, judged = [], set()
        for q in pool:
            picked.append(q); judged |= set(q.rel)
            if len(judged) >= cfg.scale_judged_budget:
                break
        qs = picked
        log(f"scale queries sampled to a {cfg.scale_judged_budget}-doc pin budget: "
            f"{len(qs)} queries, {len(judged)} judged docs", indent=1)
    noans = [q for q in load_queries(store, cfg.eval_provenance) if not q.answerable]
    out: dict[str, Any] = {"scale": [], "distractors": {}, "hard_checks": {},
                           "abstention": {}}
    cut = int(len(qs) * 0.7)
    dev, rep = qs[:cut], qs[cut:]
    thr = {nm: (calibrate_abstain(store, SCORERS[nm](), embedder, dev, cfg, tracer)
                if cfg.abstain else None)
           for nm in (cfg.baseline, cfg.candidate)}

    with store.conn.cursor() as cur:
        cur.execute("SELECT count(*) n FROM cards")
        total = cur.fetchone()["n"]
    points = [n for n in cfg.p["scale"] if n <= total] or [total]

    keep = frozenset({c for q in qs for c in q.rel})
    log(f"  {len(keep)} judged cards are pinned into every scale point; the rest of the "
        f"corpus is sampled,\n  so growing the record adds distractors instead of "
        f"deleting answers.\n")
    log(f"  {'corpus':>8}  {'scorer':<18} {'nDCG@k':>8} {'MRR':>7} {'bpref':>7}")
    for n in points:
        runs = []
        for nm in [cfg.baseline, cfg.candidate]:
            runs.append(execute_run(store, SCORERS[nm](), embedder, qs, "planted",
                                    n, cfg.topk, tracer, keep=keep))
        score_runs(runs, qs, cfg.topk)
        for r in runs:
            vals = [v for v in r.per_query.values()]
            m = {k: statistics.fmean([v[k] for v in vals if not math.isnan(v[k])])
                 for k in ("ndcg", "mrr", "bpref")}
            out["scale"].append(dict(n=n, scorer=r.scorer, **m))
            log(f"  {n:>8}  {r.scorer:<18} {m['ndcg']:>8.3f} {m['mrr']:>7.3f} "
                f"{m['bpref']:>7.3f}")
        tracer.event("scale_point", input=dict(n=n),
                     output={r.scorer: statistics.fmean(
                         [v["ndcg"] for v in r.per_query.values()
                          if not math.isnan(v["ndcg"])]) for r in runs})

    # -- what kind of wrong answer wins when the scorer is wrong
    latest = max(q.as_of for q in qs)
    rows = store.snapshot(latest)
    with store.conn.cursor() as cur:
        cur.execute("SELECT id, topic, proofline_id, is_current FROM cards")
        meta = {r["id"]: r for r in cur.fetchall()}
    for nm in [cfg.baseline, cfg.candidate]:
        r = execute_run(store, SCORERS[nm](), embedder, qs, "planted", None,
                        cfg.topk, tracer, threshold=thr[nm])
        tax: dict[str, int] = defaultdict(int)
        wrong = 0
        for q in qs:
            top = r.ranked.get(q.id, [])
            if not top or q.rel.get(top[0], 0) > 0:
                continue
            wrong += 1
            c, srcm = meta.get(top[0]), meta.get(q.source_card or "")
            if c is None:
                tax["unknown"] += 1
            elif not c["is_current"]:
                tax["superseded_version"] += 1
            elif srcm and c["proofline_id"] == srcm["proofline_id"]:
                tax["sibling_in_same_proofline"] += 1
            elif srcm and c["topic"] != srcm["topic"]:
                tax["cross_project_lookalike"] += 1
            else:
                tax["other"] += 1
        out["distractors"][nm] = dict(wrong_at_1=wrong, of=len(qs), taxonomy=dict(tax))
        log(f"\n  {nm}: wrong at rank 1 on {wrong}/{len(qs)} queries")
        for kk, vv in sorted(tax.items(), key=lambda x: -x[1]):
            log(f"      {kk:<30} {vv:>5}  ({vv/max(1,wrong):.0%} of the misses)")
        out["hard_checks"][nm] = hard_checks(r, rows, list(qs) + list(noans), cfg.topk)

    log("\n  hard checks (one occurrence blocks a release; never averaged):")
    for nm, hc in out["hard_checks"].items():
        for cname, res in hc.items():
            flag = "FAIL" if res["fails"] else "pass"
            log(f"    {nm:<18} {cname:<32} {res['fails']:>4} / {res['of']:<5} {flag}")

    # -- abstention, compared at a matched operating point
    if noans:
        log(f"\n  abstention on {len(noans)} known no-answer queries.")
        log("  Comparing two scorers at their own default thresholds compares "
            "thresholds, not scorers,\n  so both are held at a matched "
            "false-abstention rate of 10% on answerable queries.")
        for nm in [cfg.baseline, cfg.candidate]:
            sc = SCORERS[nm]()
            t = thr[nm]
            rn = execute_run(store, sc, embedder, noans, "planted", None, cfg.topk, tracer)
            rr = execute_run(store, sc, embedder, rep, "planted", None, cfg.topk, tracer)
            if t is None:
                continue
            far = sum(1 for v in rn.top_score.values() if v >= t) / max(1, len(rn.top_score))
            fabr = sum(1 for v in rr.top_score.values() if v < t) / max(1, len(rr.top_score))
            out["abstention"][nm] = dict(threshold=t, false_answer_rate=far,
                                         false_abstention_rate_heldout=fabr,
                                         target_false_abstention=cfg.abstain_fabr,
                                         n=len(noans))
            log(f"    {nm:<18} thr={t:>8.3f}  answers a no-answer query {far:>6.1%} "
                f"| abstains on an answerable one {fabr:>5.1%} (target "
                f"{cfg.abstain_fabr:.0%}, held out)")
        keys = list(out["abstention"])
        if len(keys) == 2:
            n = len(noans)
            p1 = out["abstention"][keys[0]]["false_answer_rate"]
            p2 = out["abstention"][keys[1]]["false_answer_rate"]
            need = math.ceil(7.85 * (p1 * (1 - p1) + p2 * (1 - p2)) / max(1e-9, (p1 - p2) ** 2))
            out["abstention"]["power"] = dict(n=n, observed_gap=p2 - p1, n_required=need)
            log(f"    gap = {p2-p1:+.1%} on n={n}.  To call a gap that size at 80% power "
                f"you need ~{need} no-answer queries.")
            log(f"    -> {'ENOUGH' if n >= need else 'NOT ENOUGH, this number cannot support a claim'}")
    return out


def branch4_loop(store: Store, cfg: Config, embedder: Embedder,
                 tracer: Tracer) -> dict:
    """B4: Run it every day: the comparison table and the release gate."""
    rule("B4  RUN IT EVERY DAY: the table, the slices, the power, the gate")
    qs_all = load_queries(store, cfg.eval_provenance)
    qs = [q for q in qs_all if q.answerable]
    if not qs:
        log(f"  no answerable queries with provenance '{cfg.eval_provenance}'; "
            f"no comparison possible.\n  (--source arxiv defaults to dag_mined; "
            f"--eval-provenance overrides it.)")
        return {}
    noans = [q for q in qs_all if not q.answerable]

    with store.conn.cursor() as cur:
        cur.execute("SELECT id, proofline_id FROM cards")
        pl_of = {r["id"]: r["proofline_id"] for r in cur.fetchall()}

    # Hold out by proofline, not by query. Card FEQCHS is right that the leak is at the
    # level of the record: two cards from the same line share vocabulary and structure,
    # so a random query split lets the dev set tell you about the test set.
    lines = sorted({pl_of.get(q.source_card or "", "?") for q in qs})
    rng = random.Random(cfg.seed)
    held = set(rng.sample(lines, k=max(1, len(lines) // 5)))
    dev_q = [q for q in qs if pl_of.get(q.source_card or "", "?") not in held]
    dev = [q.id for q in dev_q]
    tst = [q.id for q in qs if pl_of.get(q.source_card or "", "?") in held]

    thr = {nm: (calibrate_abstain(store, SCORERS[nm](), embedder, dev_q, cfg, tracer)
                if cfg.abstain else None)
           for nm in (cfg.baseline, cfg.candidate)}
    if cfg.abstain:
        log("  abstention thresholds calibrated on the dev prooflines only: " +
            ", ".join(f"{k}={v:.3f}" for k, v in thr.items()))

    runs = []
    for nm in [cfg.baseline, cfg.candidate]:
        runs.append(execute_run(store, SCORERS[nm](), embedder, qs, "planted",
                                None, cfg.topk, tracer, threshold=thr[nm]))
    score_runs(runs, qs, cfg.topk)
    A, B = runs

    rows: list[dict] = []

    def row(label: str, qids: Sequence[str]) -> dict:
        d, keep = paired(A, B, "ndcg", qids)
        mean, lo, hi = bootstrap_ci(d, cfg.bootstrap)
        pw = paired_power(d, cfg.target_effect); pw.pop("n", None)
        a_mean = statistics.fmean([A.per_query[q]["ndcg"] for q in keep
                                   if not math.isnan(A.per_query[q]["ndcg"])]) if keep else float("nan")
        b_mean = statistics.fmean([B.per_query[q]["ndcg"] for q in keep
                                   if not math.isnan(B.per_query[q]["ndcg"])]) if keep else float("nan")
        r = dict(label=label, n=len(d), a=a_mean, b=b_mean, mean=mean, lo=lo, hi=hi,
                 verdict=verdict_of(mean, lo, hi), **pw)
        rows.append(r)
        return r

    overall = row("overall", [q.id for q in qs])
    row("  dev split", dev)
    row("  held-out prooflines", tst)
    slice_names = sorted({s for q in qs for s in q.slices})
    for s in slice_names:
        row(f"  {s}", [q.id for q in qs if s in q.slices])

    log(f"\n  A = {cfg.baseline}   B = {cfg.candidate}   metric = condensed nDCG@{cfg.topk}"
        f"   paired, {cfg.bootstrap} bootstrap resamples\n")
    log(f"  {'':<26}{'n':>5} {'A':>7} {'B':>7} {'Δ':>8} {'95% CI':>19} "
        f"{'MDE':>7} {'n for '+str(cfg.target_effect):>10}  read")
    for r in rows:
        ci = f"[{r['lo']:+.3f},{r['hi']:+.3f}]" if not math.isnan(r["lo"]) else "        n/a"
        log(f"  {r['label']:<26}{r['n']:>5} {r['a']:>7.3f} {r['b']:>7.3f} "
            f"{r['mean']:>+8.3f} {ci:>19} {r['mde']:>7.3f} {r['n_required']:>10}  "
            f"{r['verdict']}")

    # hard checks on the candidate
    latest = max(q.as_of for q in qs_all)
    snap = store.snapshot(latest)
    cand_run = execute_run(store, SCORERS[cfg.candidate](), embedder, qs_all,
                           "planted", None, cfg.topk, tracer,
                           threshold=thr[cfg.candidate])
    hc = hard_checks(cand_run, snap, qs_all, cfg.topk)
    log("\n  hard checks on B (one occurrence blocks):")
    for cname, res in hc.items():
        log(f"    {cname:<34} {res['fails']:>4} / {res['of']:<5} "
            f"{'FAIL' if res['fails'] else 'pass'}")

    # ---- the release rule -----------------------------------------------------
    reasons: list[str] = []
    blocked = False
    for cname, res in hc.items():
        if res["fails"]:
            blocked = True
            reasons.append(f"hard check failed: {cname} ({res['fails']} occurrences)")
    for r in rows:
        if r["label"].strip() in ("overall", "dev split", "held-out prooflines"):
            continue
        if not math.isnan(r["hi"]) and r["hi"] < cfg.slice_tolerance:
            blocked = True
            reasons.append(f"slice '{r['label'].strip()}' CI entirely below "
                           f"{cfg.slice_tolerance}")
    default_ok = (not math.isnan(overall["lo"])) and overall["lo"] > 0
    under = [r["label"].strip() for r in rows
             if r["verdict"] == "UNDERPOWERED" and r["label"].startswith("  ")]

    decision = "BLOCK" if blocked else ("SHIP AS DEFAULT" if default_ok
                                        else "SHIP BEHIND A FLAG / DO NOT DEFAULT")
    rule("RELEASE DECISION", ch="=")
    log(f"  {decision}")
    for r in reasons:
        log(f"    - {r}")
    if not default_ok and not blocked:
        log(f"    - overall CI lower bound {overall['lo']:+.3f} does not clear 0: the "
            f"improvement is not established")
    if under:
        log(f"    - underpowered, reported but NOT used as gates: {', '.join(under)}")
        log(f"      (a slice that cannot resolve {cfg.target_effect} is a diagnostic, "
            f"not a decision)")
    log("")
    return dict(rows=rows, hard_checks=hc, decision=decision, reasons=reasons,
                underpowered=under, held_out_prooflines=len(held))

# ----------------------------------------------------------------------------------
# §12  CLI
# ----------------------------------------------------------------------------------

def cmd_seed(store: Store, cfg: Config, embedder: Embedder, tracer: Tracer) -> Corpus:
    p = cfg.p
    if cfg.source == "arxiv":
        log(f"loading real record from arXiv: target {p['cards']} cards")
        corpus = build_corpus_arxiv(cfg, p["cards"])
    else:
        log(f"generating synthetic record: {p['cards']} cards / "
            f"{p['prooflines']} prooflines")
        corpus = build_corpus(cfg, p["cards"], p["prooflines"])
    store.init(reset=True)
    store.load_corpus(corpus)
    for k, v in corpus.stats.items():
        log(f"  {k:<24} {v}", indent=1)
    tracer.event("seed", output=corpus.stats)
    return corpus


def cmd_ingest(store: Store, cfg: Config, embedder: Embedder, tracer: Tracer) -> None:
    with store.conn.cursor() as cur:
        cur.execute("SELECT id, txt FROM cards ORDER BY id")
        rows = cur.fetchall()
    log(f"embedding {len(rows)} cards with '{embedder.name}' ...")
    with tracer.span("embed-corpus", as_type="embedding", model=embedder.name,
                     input=dict(cards=len(rows))) as obs:
        t0 = time.perf_counter()
        mat = embedder.encode([r["txt"] for r in rows])
        store.set_embeddings([r["id"] for r in rows], mat)
        el = time.perf_counter() - t0
        Tracer.set(obs, output=dict(cards=int(mat.shape[0]), dim=int(mat.shape[1]),
                                    seconds=round(el, 2)))
    log(f"  {mat.shape[0]} x {mat.shape[1]} in {el:.1f}s "
        f"({mat.shape[0]/max(el,1e-9):.0f} cards/s)")
    n = build_usage_mined(store, cfg, embedder, tracer)
    log(f"  simulated click log -> {n} usage-mined queries")
    mh = filter_disconnected(store, cfg, embedder)
    if any(c["kept"] + c["dropped"] for f, c in mh.items() if f in LAST_HOP_FAMILIES):
        log("  disconnection filter (a single hop finds the answer -> dropped): " + ", ".join(
            f"{f} {c['dropped']}/{c['kept'] + c['dropped']}" for f, c in mh.items()
            if f in LAST_HOP_FAMILIES) + f"; pair members dropped with them {mh['pairs_dropped']}")
    tracer.event("disconnection-filter", output=mh)


# Verb-first and free of run-specific values: evaluators, dashboard filters and
# saved views target observations BY NAME, so a name carrying a timestamp or a
# scorer would silently stop matching on the next run.
BRANCH_TRACE = {"b1": "compare-label-sources", "b2": "calibrate-judge",
                "b3": "measure-scale-and-abstention", "b4": "gate-release"}


def headline(d: dict, prefix: str = "", depth: int = 0) -> dict:
    """Scalars only, dotted keys, three levels deep.

    The trace-level input/output is what the tracing table shows and what dataset
    experiments compare across runs, so it has to be the headline numbers. The full
    branch result goes to the JSON report; a raw blob here makes the table useless.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if v is None or isinstance(v, (int, float, str, bool)):
            out[key] = v
        elif isinstance(v, dict) and depth < 2:
            out.update(headline(v, f"{key}.", depth + 1))
    return out


def write_report(cfg: Config, payload: dict) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = cfg.out_dir / f"report-{stamp}.json"

    def default(o):
        if isinstance(o, (datetime,)):
            return o.isoformat()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, tuple):
            return list(o)
        return str(o)

    p.write_text(json.dumps(payload, indent=2, default=default))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    # Windows pipes default to cp1252, which cannot print the rule characters. Never let
    # the console's encoding decide whether a run completes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        prog="proofline_eval.py",
        description="Retrieval evaluation harness for a linked research record.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples
  uv run proofline_eval.py all --profile quick
  uv run proofline_eval.py all --profile full --embedder sentence-transformers
  uv run proofline_eval.py all --judge openai,anthropic --judge-pairs 80
  uv run proofline_eval.py b4 --baseline bm25 --candidate hybrid_rrf_fresh
  uv run proofline_eval.py b1 --candidate graph_boost      # watch it get refused

scorers: """ + ", ".join(SCORERS) + """
""")
    ap.add_argument("cmd", choices=["all", "seed", "ingest", "b1", "b2", "b3", "b4",
                                    "scorers", "reset"])
    ap.add_argument("--profile", default="quick", choices=list(PROFILES))
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "arxiv"],
                    help="synthetic planted corpus, or the real arXiv record")
    ap.add_argument("--eval-provenance", default=None,
                    help="query set B2/B3/B4 evaluate on "
                         "(default: planted for synthetic, generated for arxiv)")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--embedder", default="hash",
                    choices=["hash", "sentence-transformers", "openai", "specter2"])
    ap.add_argument("--judge", default="mock",
                    help="comma separated: mock,openai,anthropic")
    ap.add_argument("--judge-pairs", type=int, default=60)
    ap.add_argument("--baseline", default="bm25", choices=list(SCORERS))
    ap.add_argument("--candidate", default="hybrid_rrf", choices=list(SCORERS))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--max-eval-queries", type=int, default=1500)
    ap.add_argument("--scale-judged-budget", type=int, default=2000)
    ap.add_argument("--target-effect", type=float, default=0.02,
                    help="smallest nDCG delta worth shipping; drives the power numbers")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--no-langfuse", action="store_true")
    ap.add_argument("--no-entities", action="store_true",
                    help="skip the planted entity layer: reproduces the pre-P1 synthetic corpus")
    ap.add_argument("--fresh", action="store_true",
                    help="re-seed and re-ingest before running")
    a = ap.parse_args(argv)

    # The synthetic corpus is the only one with planted truth to evaluate against.
    # `generated` lifts words straight out of the document: a known-item smoke test
    # that a lexical scorer wins by construction. `dag_mined` -- title in, cited
    # papers out -- is the actual retrieval task, so it is what B2/B3/B4 evaluate.
    ev = a.eval_provenance or ("dag_mined" if a.source == "arxiv" else "planted")
    cfg = Config(profile=a.profile, source=a.source, eval_provenance=ev,
                 seed=a.seed, embedder=a.embedder,
                 judges=tuple(x.strip() for x in a.judge.split(",") if x.strip()),
                 judge_pairs=a.judge_pairs, topk=a.topk, bootstrap=a.bootstrap,
                 max_eval_queries=a.max_eval_queries,
                 scale_judged_budget=a.scale_judged_budget,
                 target_effect=a.target_effect, baseline=a.baseline,
                 candidate=a.candidate, database_url=a.database_url,
                 langfuse=not a.no_langfuse, entities=not a.no_entities)

    if a.cmd == "scorers":
        rule("SCORERS")
        for n, c in SCORERS.items():
            log(f"  {n:<20} graph={'yes' if c.uses_graph else 'no ':<4} {c.description}")
        rule("LABEL SOURCES")
        for n, d in [
            ("planted", "the truth. Only exists because the corpus is synthetic. The yardstick."),
            ("dag_mined", "query = a card's one-liner, relevance = the links its author recorded."),
            ("generated", "a query written from a piece, in the piece's own words. Known-item."),
            ("usage_mined", "a simulated click log. Biased toward whatever the incumbent showed."),
        ]:
            log(f"  {n:<20} {d}")
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(cfg)
    store = Store(cfg)
    if a.cmd == "reset":
        store.init(reset=True)
        log("database reset")
        return 0

    embedder = make_embedder(cfg)
    rule(f"proofline_eval  profile={cfg.profile}  embedder={embedder.name}  "
         f"A={cfg.baseline}  B={cfg.candidate}", ch="=")

    n_cards = n_emb = 0
    compatible = False
    with store.conn.cursor() as cur:
        try:
            cur.execute("SELECT count(*) n, count(embedding) e FROM cards")
            r = cur.fetchone()
            # A DB seeded before edges carried valid_from is an older schema, not a
            # corpus: re-seed it rather than half-read it.
            cur.execute("SELECT valid_from FROM edges LIMIT 0")
            cur.execute("SELECT 1 FROM true_mentions LIMIT 0")
            cur.execute("SELECT intent, family, answer, pair_id, dropped FROM queries LIMIT 0")
            cur.execute("SELECT slot FROM qrels LIMIT 0")
            n_cards, n_emb, compatible = r["n"], r["e"], True
        except psycopg.Error:
            # a table left over from an older schema is not a corpus; start clean
            store.conn.rollback()

    need_seed = a.fresh or not compatible or n_cards == 0
    if a.cmd in ("all", "seed") or need_seed:
        cmd_seed(store, cfg, embedder, tracer)
        need_ingest = True
    else:
        need_ingest = n_emb == 0
    if a.cmd in ("all", "ingest") or need_ingest:
        cmd_ingest(store, cfg, embedder, tracer)
    if a.cmd in ("seed", "ingest"):
        return 0

    assert_no_truth_leak(store)

    payload: dict[str, Any] = dict(
        config={k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(cfg).items()},
        started=datetime.now(timezone.utc).isoformat())
    todo = ["b1", "b2", "b3", "b4"] if a.cmd == "all" else [a.cmd]
    fn = {"b1": branch1_labels, "b2": branch2_judge,
          "b3": branch3_scale, "b4": branch4_loop}
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    with tracer.session(run_id):
        for b in todo:
            with tracer.span(BRANCH_TRACE[b],
                             input=dict(profile=cfg.profile, baseline=cfg.baseline,
                                        candidate=cfg.candidate,
                                        embedder=embedder.name)) as root:
                payload[b] = fn[b](store, cfg, embedder, tracer)
                Tracer.set(root, output=(headline(payload[b])
                                         or {"sections": list(payload[b])}))

    p = write_report(cfg, payload)
    tracer.flush()
    rule("", ch="=")
    log(f"report written: {p}")
    if not tracer.enabled:
        log("langfuse was not connected; set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "in .env to trace runs.")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
