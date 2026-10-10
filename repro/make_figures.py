"""Generate the figures used in the reproduction README."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEPS_PER_EPOCH = 2682


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder holding the CSVs")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    val = read_csv(os.path.join(a.dir, "metrics_val_l1.csv"))
    tr = read_csv(os.path.join(a.dir, "metrics_train_l1.csv"))
    ev = read_csv(os.path.join(a.dir, "eval_results_xs_1500_d0_clip1.csv"))
    cmp_rows = read_csv(os.path.join(a.dir, "eval_vs_official_xs.csv"))

    ep = np.array([int(r["epoch"]) for r in val])
    vl = np.array([float(r["val_l1"]) for r in val])
    ts = np.array([int(r["step"]) for r in tr])
    tl = np.array([float(r["train_l1"]) for r in tr])
    t_ep = ts / STEPS_PER_EPOCH
    win = 200
    tl_s = np.convolve(tl, np.ones(win) / win, mode="valid")
    te_s = t_ep[win - 1:]

    new_bp = np.array([float(r["new_bpppc"]) for r in cmp_rows])
    base_bp = np.array([float(r["base_bpppc"]) for r in cmp_rows])
    diff = new_bp - base_bp

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 160, "savefig.bbox": "tight"})

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.6))

    ax[0, 0].plot(ep, vl, lw=0.9, color="#1f77b4")
    i_best = int(np.argmin(vl))
    ax[0, 0].scatter([ep[i_best]], [vl[i_best]], s=28, color="#d62728", zorder=5,
                     label="best %.6f @ epoch %d" % (vl[i_best], ep[i_best]))
    ax[0, 0].set_xlabel("epoch")
    ax[0, 0].set_ylabel("val $\\ell_1$ (normalized)")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("(a) validation loss (log scale), 1500 epochs")
    ax[0, 0].legend(frameon=False, fontsize=8)

    ax[0, 1].plot(te_s, tl_s, lw=1.1, color="#2ca02c")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("epoch")
    ax[0, 1].set_ylabel("train $\\ell_1$ (window %d)" % win)
    ax[0, 1].set_title("(b) training loss (log scale)")

    bins = np.linspace(min(base_bp.min(), new_bp.min()),
                       max(base_bp.max(), new_bp.max()), 60)
    ax[1, 0].hist(base_bp, bins=bins, alpha=0.6, label="official XS (5.6492)", color="#7f7f7f")
    ax[1, 0].hist(new_bp, bins=bins, alpha=0.6, label="ours xs_1500 (5.7655)", color="#1f77b4")
    ax[1, 0].axvline(base_bp.mean(), color="#7f7f7f", ls="--", lw=1)
    ax[1, 0].axvline(new_bp.mean(), color="#1f77b4", ls="--", lw=1)
    ax[1, 0].set_xlabel("bpppc (per image)")
    ax[1, 0].set_ylabel("# images")
    ax[1, 0].set_title("(c) per-image lossless rate (n=1140)")
    ax[1, 0].legend(frameon=False, fontsize=8)

    ax[1, 1].hist(diff, bins=50, color="#d62728", alpha=0.8)
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].axvline(diff.mean(), color="#d62728", ls="--", lw=1,
                     label="mean %+.4f" % diff.mean())
    ax[1, 1].set_xlabel("bpppc(ours) - bpppc(official XS)")
    ax[1, 1].set_ylabel("# images")
    ax[1, 1].set_title("(d) paired difference: worse on %.1f%% of images"
                       % (100 * (diff > 0).mean()))
    ax[1, 1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    p1 = os.path.join(a.outdir, "reproduction_overview.png")
    fig.savefig(p1)
    plt.close(fig)
    print("saved", p1)

    # ---- benchmark comparison ----
    labels = ["CCSDS-123\n0-B-2", "RWA", "LineRWKV\nXS (paper)", "ours\nxs_1500",
              "LineRWKV S", "LineRWKV M", "LineRWKV L"]
    values = [5.801, 5.772, 5.647, 5.7655, 5.521, 5.510, 5.370]
    colors = ["#8c8c8c", "#8c8c8c", "#1f77b4", "#d62728", "#aec7e8", "#aec7e8", "#aec7e8"]
    fig2, ax2 = plt.subplots(figsize=(7.4, 3.4))
    bars = ax2.bar(labels, values, color=colors, width=0.62)
    for b, v in zip(bars, values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.3f" % v,
                 ha="center", va="bottom", fontsize=8)
    ax2.axhline(5.801, color="k", ls=":", lw=0.8)
    ax2.set_ylim(5.2, 5.95)
    ax2.set_ylabel("lossless rate (bpppc)")
    ax2.set_title("HySpecNet-11k hard test set, lossless (lower is better)")
    ax2.grid(axis="x", visible=False)
    fig2.tight_layout()
    p2 = os.path.join(a.outdir, "benchmark_comparison.png")
    fig2.savefig(p2)
    plt.close(fig2)
    print("saved", p2)


if __name__ == "__main__":
    main()
