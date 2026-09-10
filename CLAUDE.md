# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo

One file: [proofline_eval.py](proofline_eval.py) (~2900 lines). A retrieval-eval harness
that answers "which of two scorers is better, on a record nobody labelled". No package,
no tests dir, no build. Docs: [README.md](README.md) (usage + the positions it takes),
`docs/HARNESS-NOTES.md` (why the stack; gitignored, local only).

Not a git repo. Both docs reference an old path (`~/ai-agents-lab/proofline-eval`); the
file actually lives here.

## Commands

```bash
uv run proofline_eval.py all --profile smoke     # ~2s,  400 cards, does it work
uv run proofline_eval.py all --profile quick     # ~7s,  2500 cards, the default
uv run proofline_eval.py all --profile full      # ~50s, 12000 cards, 3 scale points

uv run proofline_eval.py b1                      # one branch, reusing seeded data
uv run proofline_eval.py scorers                 # registry: scorers + label sources
uv run proofline_eval.py reset                   # DROP the tables
uv run proofline_eval.py seed | ingest           # stages, run separately
```

`uv` reads the PEP 723 header and fetches Python 3.12 + 5 deps into a throwaway env;
`pgserver` boots a private Postgres under `.proofline/pgdata`. Nothing to install.

- `all` **always** re-seeds and re-ingests (`store.init(reset=True)`). `b1`–`b4` reuse
  what is in the DB and only seed if the tables are missing or empty; `--fresh` forces it.
- No test suite. The check is a run: `smoke` for a code path, `quick` for numbers.
  `assert_no_truth_leak()` runs before every branch and is the closest thing to a test.
- `--embedder sentence-transformers` downloads ~90MB on first use; `--judge openai,anthropic`
  costs real money (~$0.50 at `--judge-pairs 80`). Both keys come from `.env` only: nothing reads a key from argv.
- Reports: `.proofline/reports/report-<UTC>.json`, one per run, containing everything printed.
- `PROOFLINE_STATE` relocates `.proofline/`; `DATABASE_URL` replaces the embedded server
  (needed on linux-aarch64, where `pgserver` has no wheel).

## Layout

Sections are numbered `§0`–`§12` in comment banners; grep those to navigate.

```
§0  Config      PROFILES dict + Config dataclass (every CLI flag lands here)
§1  Corpus      build_corpus(), synthetic record with PLANTED ground truth
§1b Real corpus build_corpus_arxiv(), arXiv metadata + ogbn-arxiv citations
§2  Storage     SCHEMA + Store: snapshot(), subset(), COPY-based bulk load
§3  Embeddings  hash (offline) | sentence-transformers | openai | specter2
§4  Index       one Index per corpus size; snapshots are a boolean mask, not a rebuild
§5  Scorers     SCORERS registry, check_circularity(), CrossEncoderRerank
§6  Metrics     ndcg_at_k(condensed=), bpref, mrr, recall
§7  Statistics  bootstrap_ci, paired_power, mcnemar, cohen_kappa, kendall_tau
§8  Judges      MockJudge (simulator w/ injected bias) | HTTPJudge (openai/anthropic)
§9  Tracing     Tracer, Langfuse shim, no-ops without keys, survives v2/v3 split
§10 Harness     execute_run → score_runs (pooled) → paired; hard_checks
§11 Branches    branch1_labels .. branch4_loop, one per experiment
§12 CLI         cmd_seed, cmd_ingest, write_report, main
```

Flow: `main` → seed (§1→§2) → ingest (§3 embeddings + simulated click log) →
`branchN(store, cfg, embedder, tracer)` → JSON report. Each branch returns a dict that
becomes `payload[bN]`; anything printed must also be in that dict.

## Invariants: these are the point of the file, not incidental

Breaking one of these silently invalidates every number downstream.

1. **The section banner strings are load-bearing.** `assert_no_truth_leak()` slices the
   source between the literals `"# §4  Retrieval index"` and `"# §11  Branch"` and exits
   if `true_support` appears in that span. Renaming either banner (spacing included)
   breaks the guard silently. Nothing on the retrieval/scoring path may read
   `edges.kind = 'true_support'`.
2. **Paired, never two means.** Comparisons go through `paired()` + `bootstrap_ci()` over
   the same query ids. `verdict_of()` returns BETTER / WORSE / **UNDERPOWERED**. The
   third is a real verdict, not a fallback.
3. **Condensed nDCG + bpref, not `recall@k`.** Mined judgments are incomplete, so
   unjudged ≠ irrelevant. `score_runs()` pools the judgment set across *all* runs before
   scoring anything, which is why metrics are a second pass and not computed inline.
4. **Every query is answered against its own `as_of` snapshot**, applied as
   `idx.snapshot_mask(q.as_of)` inside the scorer. A scorer that skips the mask will
   score well and be wrong. Never rebuild the index per timestamp (that was the 2-minute
   bug; see HARNESS-NOTES).
5. **Scale points pin the needles.** `build_index(store, n_limit, keep)` passes `keep` =
   every judged card into `Store.subset()`, so growing the corpus adds distractors
   instead of deleting answers.
6. **Circularity is machine-checked.** A scorer sets `uses_graph = True` to read
   `idx.graph`/`idx.indeg`; `check_circularity()` then refuses it against any label
   source in `GRAPH_DERIVED_LABELS` (`dag_mined`). `graph_boost` exists only so the
   guard has something to refuse.
7. **Hard checks block on one occurrence** and are never averaged into a metric:
   superseded card above its successor, a result outside the query's snapshot, a
   non-empty answer to a known no-answer query. Supersession is fixed in the *candidate
   set* (`hybrid_rrf_fresh` filters, does not down-weight), not traded off in ranking.
8. **The judge is an instrument.** Pairwise only, every item run twice with the order
   swapped, disagreement under swap forced to TIE. Items where both scorers returned the
   same top-5 are skipped and counted. `--judge mock` is a bias simulator: its numbers
   are evidence about the harness, never about a judge.
9. **Abstention is a binary decision at a matched false-abstention rate**
   (`calibrate_abstain`, `cfg.abstain_fabr`), fitted on a dev split and reported held-out.
   B4 holds out **by proofline**, not by query: the leak is at the record level.

## Real data (arXiv)

`--source arxiv` loads a real corpus instead of generating one, from two cached files
under `.proofline/data/` that `prepare_arxiv.py` builds. See the README for the fetch
commands. One card per paper *version*, chained by `supersedes_id`; `proofline_id` is
the arXiv category; `true_support` is empty because nothing is planted.

Three things that bite:

- **`--source arxiv` is required on every branch run**, not just the seed. Without it
  `eval_provenance` defaults to `planted`, which real data has none of, and B2/B3/B4
  evaluate **zero queries**. They now say so instead of returning an empty table.
- **`--embedder` must match what the corpus was embedded with.** A 256-dim query
  vector against a 768-dim index raises; two models sharing a dimension would silently
  produce garbage instead.
- **B1 loses `planted`.** Link fidelity and rank bias report NOT MEASURABLE rather
  than printing zeros, because a measurement that was never taken must not look like a
  result of zero.

Extra scripts, each a PEP 723 script that imports the harness:

```
prepare_arxiv.py      join the two arXiv sources into the loader's cached inputs
branch_a_anchor.py    grade label sources by verdict agreement against a judge anchor
judge_noise_floor.py  same pair twice in the SAME order, to separate a judge's
                      sampling noise from its position bias
RESULTS.html          findings from the full four-branch run on 60k arXiv cards
```

## Extending

- **New scorer**: subclass `Scorer` in §5, set `name`/`description`/`uses_graph`,
  implement `run(q, k)` starting from `self.idx.snapshot_mask(q.as_of)`, then append the
  class to the list literal building `SCORERS`. `--baseline`/`--candidate` choices derive
  from that dict, so nothing else needs touching.
- **New CLI flag**: `argparse` in `main()` *and* the `Config` field *and* the `Config(...)`
  construction, three places, all in one screen. Current additions beyond the original
  set: `--source`, `--eval-provenance`, `--max-eval-queries`, `--scale-judged-budget`.
- **Judges** are specified `provider` or `provider:model`, so two models from one
  vendor run as two distinct judges (`anthropic:claude-sonnet-5`). Both provider paths
  must be asked the *same* question or cross-judge agreement measures the prompt.
- **Real data instead of synthetic**: fill `cards` and `edges` (`kind` in
  `parent`/`citation`), leave `true_support` empty. Every branch runs except B1's
  label-source comparison, which needs planted truth to grade labels against.
- `_INDEX_CACHE` is a process-global keyed on `(n_limit, keep)`. Mutating an `Index`
  in place leaks across branches.
