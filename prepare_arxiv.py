# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = []
# ///
"""Join the two public arXiv sources into the three files the loader reads.

The arXiv metadata snapshot has titles, abstracts, categories and per-version
timestamps but NO citations. ogbn-arxiv has 1.17M citation edges but keys them by MAG
paper id and ships no abstracts. Neither is sufficient alone, and there is no published
MAG-to-arXiv id mapping, so they are joined on normalised title via titleabs.tsv.

Writes into .proofline/data/ :
  mag2arxiv.json        MAG paper id -> arXiv id, for the nodes that resolved
  citations-arxiv.tsv   citation edges with BOTH endpoints resolved, in arXiv ids
  arxiv-subset.jsonl    metadata for just those papers, so seeding never rereads 4.6 GB

The join rate is reported rather than assumed: it is ~93.5% of nodes and ~90.3% of
edges, and the loss is not random -- it skews toward older and oddly-formatted records.
Branch 1 is a study of label incompleteness, so a second layer of it belongs in the
write-up, not in a footnote.
"""
import csv
import gzip
import json
import os
import re
import sys
from pathlib import Path

D = Path(os.environ.get("PROOFLINE_STATE", Path(__file__).resolve().parent / ".proofline")) / "data"
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

REQUIRED = {
    "arxiv-metadata-oai-snapshot.json": "the HuggingFace arXiv metadata snapshot",
    "titleabs.tsv": "extracted from titleabs.tsv.gz (it is a gzipped TAR)",
    "arxiv/mapping/nodeidx2paperid.csv.gz": "from the unzipped ogbn-arxiv archive",
    "arxiv/raw/edge.csv.gz": "from the unzipped ogbn-arxiv archive",
}


def main() -> int:
    missing = [f"  {D/p}  -- {why}" for p, why in REQUIRED.items() if not (D / p).exists()]
    if missing:
        print("missing inputs; see the README's fetch commands:\n" + "\n".join(missing))
        return 1

    # 1. normalised title -> MAG id, for the ogbn-arxiv nodes only (~180k, small)
    mag_by_title: dict[str, str] = {}
    rows = 0
    with (D / "titleabs.tsv").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2 or not p[0].strip().isdigit():
                continue
            rows += 1
            mag_by_title.setdefault(norm(p[1]), p[0].strip())
    print(f"titleabs rows={rows}  distinct normalised titles={len(mag_by_title)}")

    # 2. stream the 2.7M metadata records once, matching on normalised title
    mag2arxiv: dict[str, str] = {}
    total = collisions = 0
    with (D / "arxiv-metadata-oai-snapshot.json").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            m = mag_by_title.get(norm(d.get("title")))
            if not m:
                continue
            if m in mag2arxiv:
                collisions += 1          # duplicate title: ambiguous, keep the first
            else:
                mag2arxiv[m] = d["id"]
    pct = len(mag2arxiv) / max(1, len(mag_by_title))
    print(f"scanned {total} arXiv records")
    print(f"MAG -> arXiv resolved: {len(mag2arxiv)}/{len(mag_by_title)} = {pct:.1%} "
          f"({collisions} ambiguous titles skipped)")
    (D / "mag2arxiv.json").write_text(json.dumps(mag2arxiv))

    # 3. node index -> arXiv id, then edges with both endpoints resolved
    idx2arxiv: dict[int, str] = {}
    with gzip.open(D / "arxiv/mapping/nodeidx2paperid.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            a = mag2arxiv.get(row["paper id"].strip())
            if a:
                idx2arxiv[int(row["node idx"])] = a
    kept = dropped = 0
    with gzip.open(D / "arxiv/raw/edge.csv.gz", "rt") as f, \
         (D / "citations-arxiv.tsv").open("w") as out:
        for line in f:
            s, t = line.strip().split(",")
            a, b = idx2arxiv.get(int(s)), idx2arxiv.get(int(t))
            if a and b:
                out.write(f"{a}\t{b}\n"); kept += 1
            else:
                dropped += 1
    print(f"citation edges usable: {kept}/{kept+dropped} = {kept/max(1,kept+dropped):.1%} "
          f"({dropped} dropped -- NOT a random sample of the graph)")

    # 4. metadata for just the papers in the citation graph
    want = set(idx2arxiv.values())
    n = 0
    keys = ("id", "title", "abstract", "categories", "versions", "update_date")
    with (D / "arxiv-metadata-oai-snapshot.json").open(encoding="utf-8", errors="replace") as f, \
         (D / "arxiv-subset.jsonl").open("w") as out:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d["id"] in want:
                out.write(json.dumps({k: d[k] for k in keys}) + "\n"); n += 1
    print(f"metadata subset: {n} papers -> {D/'arxiv-subset.jsonl'}")
    print("\nready. seed with:\n  uv run proofline_eval.py seed --source arxiv "
          "--profile quick --embedder specter2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
