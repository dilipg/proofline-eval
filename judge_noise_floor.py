# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.26", "psycopg[binary]>=3.1",
#   "pgserver>=0.1.4; sys_platform == 'darwin' or platform_machine == 'x86_64'",
#   "httpx>=0.27", "langfuse>=4.0", "sentence-transformers>=3.0", "adapters>=1.0",
# ]
# ///
"""How much of 'swap inconsistency' is position bias, and how much is just noise?

Branch B reports the rate at which a judge's verdict flips when the two lists are
swapped, and reads that as position bias. That reading only holds if the judge is
deterministic. Ours is not: removing `temperature: 0` (Sonnet 5 rejects the parameter
outright) left every Anthropic judge sampling, and two runs over identical inputs
returned different kappas.

So this measures the NOISE FLOOR: ask the same judge the same question twice in the
SAME order. Any flip there involves no position change at all, so it is pure sampling
noise. Position bias is what survives after that is subtracted.

  swap-flip rate  =  noise  +  position effect
  same-order rate =  noise
"""
import sys, argparse
sys.path.insert(0, ".")
import proofline_eval as pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="anthropic:claude-haiku-4-5,anthropic:claude-sonnet-5")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--candidate", default="hybrid_rrf_fresh")
    ap.add_argument("--embedder", default="specter2")
    a = ap.parse_args()

    cfg = pe.Config(source="arxiv", eval_provenance="dag_mined", embedder=a.embedder,
                    baseline=a.baseline, candidate=a.candidate)
    tracer, store = pe.Tracer(cfg), pe.Store(cfg)
    embedder = pe.make_embedder(cfg)

    qs = [q for q in pe.load_queries(store, cfg.eval_provenance)
          if q.answerable][:a.pairs]
    runs = [pe.execute_run(store, pe.SCORERS[nm](), embedder, qs, "noise", None,
                           cfg.topk, tracer) for nm in (a.baseline, a.candidate)]
    with store.conn.cursor() as cur:
        cur.execute("SELECT id, title, one_line FROM cards")
        txt = {r["id"]: f"{r['title']}: {r['one_line']}" for r in cur.fetchall()}

    pe.rule("JUDGE NOISE FLOOR  (same pair, same order, asked twice)")
    out = {}
    for spec in [x for x in a.judge.split(",") if x]:
        judge = pe.HTTPJudge(spec)
        judge.tracer = tracer
        same, flips, errs, asked = 0, 0, 0, 0
        for q in qs:
            A = runs[0].ranked.get(q.id, [])[:5]
            B = runs[1].ranked.get(q.id, [])[:5]
            if A == B or not A or not B:
                continue
            IA = [(c, txt.get(c, c)[:220]) for c in A]
            IB = [(c, txt.get(c, c)[:220]) for c in B]
            v1 = judge.compare(q.text, IA, IB)
            v2 = judge.compare(q.text, IA, IB)      # identical call, no swap
            if v1 is None or v2 is None:
                errs += 1; continue
            asked += 1
            if v1 == v2:
                same += 1
            else:
                flips += 1
        rate = flips / max(1, asked)
        out[spec] = dict(asked=asked, same=same, flips=flips, errors=errs,
                         noise_rate=rate)
        pe.log(f"  {spec:<28} asked={asked}  identical-answer={same}  "
               f"flipped={flips}  -> NOISE FLOOR {rate:.1%}"
               + (f"  [{errs} errors]" if errs else ""))

    pe.log("\n  reading: subtract the noise floor from Branch B's swap-inconsistency.")
    pe.log("  What is left is attributable to position; what is not is the judge "
           "disagreeing\n  with itself, which is a different defect with a different "
           "fix (sample twice and\n  take the mode, or pin temperature where the model "
           "still allows it).")
    p = pe.write_report(cfg, {"judge_noise_floor": out})
    tracer.flush(); store.close()
    pe.log(f"\nreport written: {p}")


if __name__ == "__main__":
    main()
