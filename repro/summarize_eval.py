"""Append a test-set result section to metrics_summary.md from an eval CSV."""

import argparse
import csv
import os
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--title", default="测试集无损码率（delta=0, clip=1）")
    ap.add_argument("--run", default="")
    a = ap.parse_args()

    with open(a.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty csv: " + a.csv)

    bp = [float(r["bpppc"]) for r in rows]
    total_bits = sum(int(r["bytes_total"]) * 8 for r in rows)
    total_px = sum(int(r["rows"]) * int(r["cols"]) * int(r["bands"]) for r in rows)

    lines = ["", "## %s" % a.title, ""]
    if a.run:
        lines.append("运行：`%s`" % a.run)
        lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append("| 图像数 | %d |" % len(rows))
    lines.append("| 平均 bpppc（按像素加权） | %.4f |" % (total_bits / total_px))
    lines.append("| 平均 bpppc（逐图算术平均） | %.4f |" % statistics.mean(bp))
    lines.append("| 最小 / 最大 bpppc | %.4f / %.4f |" % (min(bp), max(bp)))
    lines.append("| 总字节数 | %d |" % sum(int(r["bytes_total"]) for r in rows))
    lines.append("")
    lines.append("逐图明细：`%s`" % os.path.basename(a.csv))
    lines.append("")

    with open(a.summary, "a") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
