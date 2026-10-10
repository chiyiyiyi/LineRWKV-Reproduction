"""Export LineRWKV training metrics from a Lightning TensorBoard event file.

Writes:
  metrics_val_l1.csv    one row per epoch (epoch, step, val_l1, wall_time)
  metrics_train_l1.csv  one row per logged training step
  train_summary.json    machine-readable summary
  metrics_summary.md    human-readable summary
"""

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone

from tensorboard.backend.event_processing import event_accumulator


def load(events):
    ea = event_accumulator.EventAccumulator(
        events, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        out[tag] = [(s.step, s.value, s.wall_time) for s in ea.Scalars(tag)]
    return out


def fmt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--wall-hours", type=float, default=0.0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    data = load(a.events)

    epochs = data.get("epoch", [])
    val = data.get("val_l1", [])
    train = data.get("train_l1", [])

    with open(os.path.join(a.outdir, "metrics_val_l1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "val_l1", "wall_time"])
        for i, (step, value, wall) in enumerate(val):
            w.writerow([i, step, "%.8g" % value, "%.0f" % wall])

    with open(os.path.join(a.outdir, "metrics_train_l1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_l1", "wall_time"])
        for step, value, wall in train:
            w.writerow([step, "%.8g" % value, "%.0f" % wall])

    n_epochs = len(val)
    t0 = min(x[2] for x in train) if train else 0.0
    t1 = max(x[2] for x in train) if train else 0.0
    train_dur_h = (t1 - t0) / 3600.0 if train else 0.0

    summary = {
        "run": a.run,
        "config": a.config,
        "epochs_logged": n_epochs,
        "train_points_logged": len(train),
        "epoch_steps_span": [epochs[0][0], epochs[-1][0]] if epochs else [],
        "start_time": fmt(t0) if t0 else "",
        "end_time": fmt(t1) if t1 else "",
        "train_wall_hours": round(train_dur_h, 2),
    }
    if val:
        vals = [v for _, v, _ in val]
        summary["val_l1"] = {
            "first": vals[0],
            "last": vals[-1],
            "best": min(vals),
            "best_epoch": int(min(range(len(vals)), key=lambda i: vals[i])),
            "mean_last_100": statistics.mean(vals[-100:]),
            "mean_all": statistics.mean(vals),
        }
    if train:
        tv = [v for _, v, _ in train]
        summary["train_l1"] = {
            "first": tv[0],
            "last": tv[-1],
            "best": min(tv),
            "mean_last_100_logged": statistics.mean(tv[-100:]),
        }

    with open(os.path.join(a.outdir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    def g(d, k):
        return d.get(k, float("nan"))

    lines = []
    lines.append("# LineRWKV 训练指标汇总\n")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append("| 训练轮数 (epochs) | %d |" % n_epochs)
    lines.append("| 训练日志点数 | %d |" % len(train))
    lines.append("| 起止时间 | %s → %s |" % (summary["start_time"], summary["end_time"]))
    lines.append("| 训练总时长 | %.2f 小时 (%.1f 天) |" % (train_dur_h, train_dur_h / 24))
    if val:
        lines.append("| val_l1 首值 | %.6f |" % vals[0])
        lines.append("| val_l1 末值 | %.6f |" % vals[-1])
        lines.append(
            "| val_l1 最优 | %.6f (epoch %d) |"
            % (summary["val_l1"]["best"], summary["val_l1"]["best_epoch"])
        )
        lines.append("| val_l1 最后 100 epoch 均值 | %.6f |" % summary["val_l1"]["mean_last_100"])
    if train:
        lines.append("| train_l1 首值 | %.6f |" % tv[0])
        lines.append("| train_l1 末值 | %.6f |" % tv[-1])
        lines.append("| train_l1 最优 | %.6f |" % summary["train_l1"]["best"])
    lines.append("")
    lines.append("> 说明：val_l1 / train_l1 都是模型在 **归一化后** 的像素尺度上的 L1 误差")
    lines.append("> （`dataset.py` 中图像整体除以 1000），因此 0.009 大约相当于 9 个 DN。")
    lines.append("")
    with open(os.path.join(a.outdir, "metrics_summary.md"), "w") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
