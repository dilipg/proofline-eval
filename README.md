# proofline-eval

**A RAG retrieval-evaluation experiment on a scientific research claims graph.**

A single-file harness for one question: **two candidate retrievers rank pieces of a
shared research record for a query. Which is better, before either goes live?**

That question is easy to answer badly. You need labels nobody wrote, a metric those
labels actually permit, a scale test that doesn't lie, and a gate that doesn't fire on
noise. This harness implements all four as runnable experiments, on a synthetic corpus
with planted ground truth *or* on 60,001 real arXiv papers.

The record under test is a **graph of scientific claims**, not a flat document dump.
Every card is one versioned claim; edges are the citations its authors actually recorded.
That shape is the whole point. It is what makes the retrieval problem hard in ways a
plain corpus is not, and it is where the labels come from:

- **Graph-derived relevance.** The citation edges *are* relevance judgments, made by
  domain experts with full context. Branch 1 measures how biased and incomplete they are
  instead of assuming either way. On real data they recover a genuinely partial view.
- **Versioned claims.** 26,355 of 60,001 cards are superseded revisions of a claim whose
  current form sits elsewhere in the corpus. Near-identical in meaning, so no embedding
  separates them; separating them is a metadata problem, and treating it as one converts
  a ranking trade-off into a guarantee.
- **Claims move.** Every query carries an `as_of` and is answered against the graph as it
  stood then, across a 30-year span. A frozen RAG test set does not merely go stale, it
  goes wrong. It penalises whichever retriever finds the *current* version of a claim.
- **Circularity is real here.** A retriever that reads graph structure cannot be scored
  against graph-derived labels. The harness refuses that comparison in code.

Retrievers compared: BM25, dense (SPECTER2, a scientific-document encoder), reciprocal
rank fusion, RRF plus a supersession filter, and a cross-encoder reranker.

**[Read the results →](RESULTS.html)**: all four branches run against the real arXiv
claims graph.

---

## Why it exists

Most retrieval evaluation measures *how good* a scorer is. That is the harder problem and
usually not the one you have. Selecting between two scorers only needs a reliable **sign
on the difference**, which is a much smaller object and reachable by methods absolute
measurement cannot use. Everything here is paired, bootstrapped, and reports its own
statistical power.

The four branches:

| | Question |
|---|---|
| **B1** | Where do labels come from, and does each source pick the same winner the truth would? |
| **B2** | Ranking metric or LLM judge, and how biased is the judge, measurably? |
| **B3** | Does the scorer survive a 10× corpus, hard distractors, and "no answer"? |
| **B4** | What runs on every change, and what blocks a bad scorer from shipping? |

Positions it takes, and enforces in code rather than convention:

- **Incomplete labels are not negative labels.** Default metric is condensed-list nDCG
  plus bpref. `recall@k` punishes the scorer that surfaces something relevant nobody
  labelled, precisely the scorer you were hoping to find.
- **Supersession is a filter problem, not a ranking one.** Two versions of a claim are
  near-identical in meaning, so no semantic scorer separates them. Doing it in the
  candidate set turns a trade-off into a guarantee: 2,532 failures to zero on real data.
- **Circularity is machine-checked.** A scorer that reads graph structure is *refused*
  against graph-derived labels, not trusted to remember.
- **`UNDERPOWERED` is a verdict.** Every row prints its CI, its minimum detectable
  effect, and the n required. A slice that cannot resolve the effect is a diagnostic.
- **Every query carries an `as_of`** and is answered against the corpus snapshot at that
  time. A frozen test set doesn't just go stale, it goes wrong.

---

## Run it

Requires [`uv`](https://docs.astral.sh/uv/). Nothing else: `uv` reads the PEP 723 header
and fetches Python plus dependencies; Postgres boots itself.

```bash
uv run proofline_eval.py all --profile quick     # synthetic, ~2.5k cards, seconds
uv run proofline_eval.py scorers                 # list scorers and label sources
```

Profiles: `smoke` (400 cards) · `quick` (2,500) · `full` (12,000) · `arxiv-scale` (60,000).

### Credentials

Copy `.env.example` to `.env` and fill in what you need. Nothing reads a key from the
command line and no key is printed.

```
ANTHROPIC_API_KEY=       # LLM judge
OPENAI_API_KEY=          # second judge, for cross-family agreement
LANGFUSE_PUBLIC_KEY=     # optional tracing, pk-lf-… not sk-lf-…
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=           # us.cloud.langfuse.com or cloud.langfuse.com, regions differ
DATABASE_URL=            # optional; unset boots a private embedded Postgres
```

---

## Real data: the arXiv claims graph

The synthetic corpus exists so the harness can grade its *own labels*. You cannot do
that on real data, which is the point of B1. But everything else runs better on the real
thing: genuine version chains, genuine citation edges, genuine incompleteness.

Two public sources are joined into one graph of scientific claims: arXiv metadata
supplies the claims and their revision history, ogbn-arxiv supplies the citation edges
between them. 60,001 cards over 33,646 papers, 898,609 citation edges, 41 subject areas,
1995–2025.

Two public sources, no Kaggle account needed. Both are anonymous downloads.

```bash
mkdir -p .proofline/data && cd .proofline/data

# 1. arXiv metadata: 4.6 GB, 2.7M records, title/abstract/categories/versions
curl -L -C - -o arxiv-metadata-oai-snapshot.json \
  https://huggingface.co/datasets/jackkuo/arXiv-metadata-oai-snapshot/resolve/main/arxiv-metadata-oai-snapshot.json

# 2. ogbn-arxiv citation graph: 79 MB, 1,166,243 edges over 169,343 CS papers
curl -L -o ogbn-arxiv.zip http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
unzip -q ogbn-arxiv.zip

# 3. MAG↔title bridge: 70 MB (a gzipped tar, despite the name)
curl -L -o titleabs.tsv.gz https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
tar -xzf titleabs.tsv.gz
cd ../..
```

Then build the two cached artifacts the loader reads. The metadata gives versions and
timestamps but no citations; ogbn-arxiv gives citations keyed by MAG id. They are joined
on normalised title:

```bash
uv run prepare_arxiv.py     # writes mag2arxiv.json, citations-arxiv.tsv, arxiv-subset.jsonl
```

Roughly 93.5% of MAG nodes resolve to arXiv ids, giving **1,053,367 usable citation
edges** over 158,160 papers. The join loss is not random: it skews toward older and
oddly-formatted records, and that matters for B1, which is a study of label
incompleteness.

After this, seeding never touches the 4.6 GB file again.

### Seed and run

```bash
uv run proofline_eval.py seed --source arxiv --profile quick --embedder specter2
uv run proofline_eval.py b1 --source arxiv --embedder specter2
uv run proofline_eval.py b4 --source arxiv --embedder specter2 --candidate rerank
```

Two flags that must match the data or you get a silent mismatch:

- `--source arxiv`: otherwise `--eval-provenance` defaults to `planted`, which real
  data does not have, and the branch evaluates **zero queries**.
- `--embedder` must be the one the corpus was embedded with. Mixing a 256-dim query
  vector into a 768-dim index raises; two models sharing a dimension would not.

One card per paper *version*, chained by `supersedes_id`. `proofline_id` is the arXiv
category. Papers held outside the sampled subset become genuine no-answer queries: their evidence really is absent.

---

## Ontology ingestion (Semantica)

The synthetic corpus plants a typed entity layer: methods, datasets, metrics, materials
and organisms. The relations between them (`extends`, `evaluated_on`, `measured_by`,
`reports`) are written into card text in several phrasings, and recorded as truth.
`ingest_semantica.py` extracts that layer back out with [Semantica](https://github.com/semantica-agi/semantica)
(pinned 0.7.0), resolves aliases, gives every fact a validity window from the version
chain, and writes it next to the cards. B1 then grades each extractor against the truth.

    uv run proofline_eval.py seed --profile quick
    uv run ingest_semantica.py --method regex                  # offline
    uv run ingest_semantica.py --method llm --extract-limit 300 # anthropic:claude-haiku-4-5
    uv run proofline_eval.py b1

The default model is Claude Haiku 4.5. `--llm anthropic:claude-opus-5` is slower and costlier.
Extraction is cached by card text, so re-seeding does not re-bill. `all` re-seeds and
empties the extracted tables, so run it before extracting, not after.

`--viz` (or `--viz-only`) draws the result into `.proofline/viz/`. Each picture is
masked at an as-of date, so it shows the record as it stood then:
- `dag.html`: the citation DAG
- `ontology.html`: cards, entities and the typed facts between them
- `ontology-classes.html`: the inferred class hierarchy
- `timeline.html`: the same neighbourhood at several dates
- `graph.json`: open with `semantica-explorer --graph .proofline/viz/graph.json`,
  after `pip install "semantica[explorer]"`

`--no-entities` reproduces the pre-ontology corpus byte for byte.

---

## Scorers

| name | graph? | what it is |
|---|---|---|
| `bm25` | no | Okapi BM25, k1=1.2 b=0.75. The incumbent. |
| `dense` | no | Cosine over embeddings, brute force. No ANN index at this scale. |
| `hybrid_rrf` | no | Reciprocal rank fusion of the two, k=60. |
| `hybrid_rrf_fresh` | no | RRF + drop superseded + mild recency decay. |
| `rerank` | no | `hybrid_rrf_fresh` top-50, reranked by a cross-encoder. |
| `graph_boost` | **yes** | RRF + in-degree boost. Exists so the circularity guard has something to refuse. |

Embedders: `hash` (offline, deterministic), `sentence-transformers`, `openai`,
`specter2` (asymmetric: documents through the `proximity` adapter, queries through
`adhoc_query`).

---

## Files

```
proofline_eval.py     the harness
prepare_arxiv.py      join the two arXiv sources into the cached loader inputs
ingest_semantica.py   ontology ingestion with Semantica (extract, resolve, visualize)
branch_a_anchor.py    grade label sources by verdict agreement against a judge anchor
judge_noise_floor.py  separate a judge's position bias from its sampling noise
RESULTS.html          findings from the full four-branch run on 60k arXiv cards
.proofline/reports/   one JSON per run; every printed number is also written here
tests/                pytest unit tests; add --with semantica[...] to run the adapter tests
```

`uv run proofline_eval.py reset` drops the tables. `--fresh` re-seeds before running.
