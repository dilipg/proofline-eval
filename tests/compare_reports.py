"""Diff the branch payloads of two harness reports; exit 1 on any unexpected change.

    python tests/compare_reports.py BASE.json NEW.json --allow b1.link_fidelity b1.rank_bias

Keys that exist only in NEW are additions (B1's new `extraction` block), not regressions.
Timing fields are ignored."""
import json
import math
import sys

IGNORE = {"seconds", "ms", "timing_ms", "started"}


def walk(a, b, path, out, allow):
    if any(path.startswith(p) for p in allow):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in IGNORE:
                continue
            if k not in b:
                out.append(f"{path}.{k}: missing in new")
            else:
                walk(a[k], b[k], f"{path}.{k}", out, allow)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out, allow)
        return
    if isinstance(a, float) and isinstance(b, float):
        if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a} -> {b}")
        return
    if a != b:
        out.append(f"{path}: {a!r} -> {b!r}")


def main(argv):
    base, new = (json.load(open(p, encoding="utf-8")) for p in argv[:2])
    allow = argv[3:] if len(argv) > 2 and argv[2] == "--allow" else []
    diffs = []
    for b in ("b1", "b2", "b3", "b4"):
        walk(base.get(b), new.get(b), b, diffs, allow)
    for d in diffs:
        print(d)
    print(f"{len(diffs)} unexpected difference(s)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
