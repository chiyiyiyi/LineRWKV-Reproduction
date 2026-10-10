"""Paired comparison of two LineRWKV compression evaluations.

Baseline rates are reconstructed from the side files left in a result folder
(same accounting as repro/compress_eval.py).
"""

import argparse
import csv
import os
import statistics


def baseline_rate(baseline_dir, name, delta, bands):
    prefix = os.path.join(baseline_dir, "%s_d%d" % (name, delta))
    paths = [prefix + "_compressed.cmp", prefix + "_numerical.bin",
             prefix + "_side_info.h5_mu.cmp", prefix + "_side_info.h5_sigma.cmp"]
    if not all(os.path.exists(p) for p in paths):
        return None
    return sum(os.path.getsize(p) for p in paths) + 64 * bands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-csv", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--delta", type=int, default=0)
    a = ap.parse_args()

    with open(a.new_csv) as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        bands = int(r["bands"])
        base = baseline_rate(a.baseline_dir, r["name"], a.delta, bands)
        if base is None:
            continue
        px = int(r["rows"]) * int(r["cols"]) * bands
        new_rate = int(r["bytes_total"])
        out.append({
            "name": r["name"],
            "new_bytes": new_rate,
            "new_bpppc": new_rate * 8 / px,
            "base_bytes": base,
            "base_bpppc": base * 8 / px,
            "diff_bpppc": (new_rate - base) * 8 / px,
            "bytes_saved": base - new_rate,
        })

    if not out:
        raise SystemExit("no overlapping images")

    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "new_bytes", "new_bpppc", "base_bytes",
                                          "base_bpppc", "diff_bpppc", "bytes_saved"])
        w.writeheader()
        for r in out:
            w.writerow({k: ("%.6f" % v if isinstance(v, float) else v) for k, v in r.items()})

    d = [r["diff_bpppc"] for r in out]
    nb = [r["new_bpppc"] for r in out]
    bb = [r["base_bpppc"] for r in out]
    print("images compared        : %d" % len(out))
    print("new   mean bpppc (pair): %.4f" % statistics.mean(nb))
    print("base  mean bpppc (pair): %.4f" % statistics.mean(bb))
    print("mean paired diff       : %+.4f bpppc" % statistics.mean(d))
    print("median paired diff     : %+.4f bpppc" % statistics.median(d))
    print("std of diff            : %.4f" % (statistics.pstdev(d) if len(d) > 1 else 0.0))
    print("worst / best diff      : %+.4f / %+.4f" % (max(d), min(d)))
    print("images better          : %d (%.1f%%)" % (sum(1 for x in d if x < 0), 100 * sum(1 for x in d if x < 0) / len(d)))
    print("images equal           : %d" % sum(1 for x in d if x == 0))
    print("images worse           : %d (%.1f%%)" % (sum(1 for x in d if x > 0), 100 * sum(1 for x in d if x > 0) / len(d)))
    print("total bytes saved      : %d (%.1f MB)" % (sum(r["bytes_saved"] for r in out),
                                                     sum(r["bytes_saved"] for r in out) / 1e6))

    worst = sorted(out, key=lambda r: -r["diff_bpppc"])[:5]
    best = sorted(out, key=lambda r: r["diff_bpppc"])[:5]
    print("\n-- biggest regressions --")
    for r in worst:
        print("  %+.4f  %s" % (r["diff_bpppc"], r["name"]))
    print("-- biggest improvements --")
    for r in best:
        print("  %+.4f  %s" % (r["diff_bpppc"], r["name"]))


if __name__ == "__main__":
    main()
