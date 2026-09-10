# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.26", "psycopg[binary]>=3.1",
#   "pgserver>=0.1.4; sys_platform == 'darwin' or platform_machine == 'x86_64'",
#   "httpx>=0.27", "langfuse>=4.0", "sentence-transformers>=3.0", "adapters>=1.0",
# ]
# ///
"""Branch A on a real record: grade each label source by VERDICT AGREEMENT.

The brief's discipline is "grade each label source by whether it picks the same winner
the anchor picked, not by how precise its labels look". On a real corpus there is no
planted truth and no human anchor, so the anchor here is a calibrated LLM judge run on
realistic queries (paper titles) with no labels at all -- it compares the two rankings
directly. Branch B calibrates that judge first; this script consumes it.

Two things this CANNOT do, stated so the numbers are not over-read:
  * A judge-anchored Branch A is weaker than a human-anchored one. The judge has its
    own measured biases and they are reported alongside, not assumed away.
  * Label sources differ in BOTH query text and relevance labels, so there is no
    "same query" across sources and no per-query kappa between them. Agreement is
    therefore at the level of the aggregate verdict, which is what the brief compares.
"""
import sys, json, math, random, argparse
sys.path.insert(0, ".")
import proofline_eval as pe


def verdict_from_prefs(prefs, n_boot=2000, seed=7):
    """Aggregate per-query judge preferences (+1 B better, -1 A better, 0 tie)."""
    if not prefs:
        return dict(n=0, mean=float("nan"), lo=float("nan"), hi=float("nan"),
                    verdict="NO DATA")
    mean, lo, hi = pe.bootstrap_ci(prefs, n=n_boot)
    return dict(n=len(prefs), mean=mean, lo=lo, hi=hi,
                verdict=pe.verdict_of(mean, lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="anthropic,openai")
    ap.add_argument("--anchor-queries", type=int, default=120)
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--candidate", default="hybrid_rrf_fresh")
    ap.add_argument("--embedder", default="specter2")
    a = ap.parse_args()

    cfg = pe.Config(source="arxiv", eval_provenance="dag_mined", embedder=a.embedder,
                    baseline=a.baseline, candidate=a.candidate,
                    judges=tuple(x for x in a.judge.split(",") if x))
    tracer, store = pe.Tracer(cfg), pe.Store(cfg)
    embedder = pe.make_embedder(cfg)
    out = {"config": dict(baseline=a.baseline, candidate=a.candidate,
                          embedder=a.embedder, anchor_queries=a.anchor_queries)}

    # ---- 1. each label source's own verdict ---------------------------------------
    pe.rule("BRANCH A  LABEL SOURCES vs A JUDGE ANCHOR (real record, no planted truth)")
    out["sources"] = {}
    for src in ["dag_mined", "generated", "usage_mined"]:
        qs = [q for q in pe.load_queries(store, src) if q.answerable]
        if not qs:
            continue
        runs = []
        refused = None
        for nm in (a.baseline, a.candidate):
            sc = pe.SCORERS[nm]()
            r = pe.check_circularity(sc, src)
            if r:
                refused = r; break
            runs.append(pe.execute_run(store, sc, embedder, qs, src, None,
                                       cfg.topk, tracer))
        if refused:
            out["sources"][src] = {"refused": refused}
            pe.log(f"  {src:<12} {refused}")
            continue
        pe.score_runs(runs, qs, cfg.topk)
        d, _ = pe.paired(runs[0], runs[1], "ndcg")
        mean, lo, hi = pe.bootstrap_ci(d, n=cfg.bootstrap)
        v = pe.verdict_of(mean, lo, hi)
        out["sources"][src] = dict(n=len(d), mean=mean, lo=lo, hi=hi, verdict=v)
        pe.log(f"  {src:<12} n={len(d):<5} d nDCG@10={pe.fmt(mean)} "
               f"[{pe.fmt(lo)},{pe.fmt(hi)}]  -> {v}")

    # ---- 2. the anchor: a judge, on titles, with no labels at all -------------------
    # Realistic query text (a title), but the verdict comes from comparing the two
    # rankings directly rather than from any label set. That is what makes it
    # independent of the label sources it is grading.
    anchor_qs = [q for q in pe.load_queries(store, "dag_mined") if q.answerable]
    rng = random.Random(cfg.seed)
    rng.shuffle(anchor_qs)
    anchor_qs = anchor_qs[: a.anchor_queries]
    runs = [pe.execute_run(store, pe.SCORERS[nm](), embedder, anchor_qs, "anchor",
                           None, cfg.topk, tracer) for nm in (a.baseline, a.candidate)]
    with store.conn.cursor() as cur:
        cur.execute("SELECT id, title, one_line FROM cards")
        txt = {r["id"]: f"{r['title']}: {r['one_line']}" for r in cur.fetchall()}

    out["anchor"] = {}
    per_judge = {}
    for jn in cfg.judges:
        judge = pe.HTTPJudge(jn)
        judge.tracer = tracer
        prefs, errs, ident, incons = [], 0, 0, 0
        for q in anchor_qs:
            A = runs[0].ranked.get(q.id, [])
            B = runs[1].ranked.get(q.id, [])
            if A[:5] == B[:5]:
                ident += 1
                continue
            d = pe.judge_duel(judge, q, A, B, lambda c: txt.get(c, c), tracer=tracer)
            if d["verdict"] is None:
                errs += 1; continue
            if not d["consistent"]:
                incons += 1
            prefs.append({"A": -1.0, "B": 1.0, "TIE": 0.0}[d["verdict"]])
        v = verdict_from_prefs(prefs)
        v.update(errors=errs, identical=ident, swap_inconsistent=incons,
                 swap_rate=incons / max(1, len(prefs)))
        per_judge[jn] = v
        out["anchor"][jn] = v
        pe.log(f"\n  anchor[{jn}]  n={v['n']}  mean pref={pe.fmt(v['mean'])} "
               f"[{pe.fmt(v['lo'])},{pe.fmt(v['hi'])}] -> {v['verdict']}"
               f"   (errors={errs} identical={ident} swap-flips={incons})")

    # ---- 3. verdict agreement -------------------------------------------------------
    pe.rule("VERDICT AGREEMENT  (does the label source pick the anchor's winner?)")
    for jn, av in per_judge.items():
        pe.log(f"\n  anchor = judge '{jn}' -> {av['verdict']}")
        for src, sv in out["sources"].items():
            if "verdict" not in sv:
                continue
            agree = "AGREES" if sv["verdict"] == av["verdict"] else "DISAGREES"
            same_sign = (sv["mean"] > 0) == (av["mean"] > 0) if av["n"] else None
            pe.log(f"    {src:<12} {sv['verdict']:<13} {agree}"
                   f"   (direction {'same' if same_sign else 'opposite'})")
    out["agreement"] = {jn: {s: (v.get("verdict") == av["verdict"])
                             for s, v in out["sources"].items() if "verdict" in v}
                        for jn, av in per_judge.items()}

    p = pe.write_report(cfg, {"branch_a_anchor": out})
    tracer.flush(); store.close()
    pe.log(f"\nreport written: {p}")


if __name__ == "__main__":
    main()
