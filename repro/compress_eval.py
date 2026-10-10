"""Evaluate LineRWKV lossless/near-lossless compression on a HySpecNet-11k split.

Extends the original repro/compress_eval.py with:
  --tag      output folder / file prefix (defaults to --config, so the original
             behaviour is unchanged when --tag is omitted)
  --resume   skip images already present in the result CSV and reuse their numbers
  --csv      where to write per-image results (default: <outdir>/eval_results_*.csv)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import rasterio

INVALID = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
           160, 161, 162, 163, 164, 165, 166]
VALID = [c + 1 for c in range(224) if c not in INVALID]

# NOTE: lab-local absolute paths were replaced by placeholders for this public copy.
# Point them at your own checkout / dataset root before running.
ROOT = "<REPO_ROOT>"
DS = "<DATASET_ROOT>"
BIN_ENCODER = os.path.join(ROOT, "Code", "compressor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="xs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model_file", default="")
    ap.add_argument("--config_file", default="")
    ap.add_argument("--difficulty", default="hard")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--delta", type=int, default=0)
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--clip", type=int, default=1)
    ap.add_argument("--workdir", default="<WORK_DIR>")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    tag = a.tag or a.config
    cfg_model = a.model_file or os.path.join(ROOT, "Results", "linerwkv", a.config, "rwkv-final.pth")
    cfg_file = a.config_file or os.path.join(ROOT, "log_dir", "linerwkv", a.config, "config.txt")
    outdir = os.path.join(ROOT, "Results", "linerwkv", tag)
    os.makedirs(outdir, exist_ok=True)

    with open(cfg_file) as f:
        cfg = json.load(f)
    print("model :", cfg_model, flush=True)
    print("config:", {k: cfg[k] for k in ("dim_enc", "dim_att", "dim_ffn",
                                          "N_layers_encoder", "n_layer_lines",
                                          "n_layer_bands", "N_layers_decoder")}, flush=True)

    csv_path = a.csv or os.path.join(outdir, "eval_results_%s_d%d_clip%d.csv" % (tag, a.delta, a.clip))
    done = {}
    if a.resume and os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done[row["name"]] = row
        print("resume: %d images already in %s" % (len(done), csv_path), flush=True)
    new_file = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    writer = csv.writer(fcsv)
    if new_file:
        writer.writerow(["name", "rows", "cols", "bands",
                         "bytes_cmp", "bytes_num", "bytes_mu", "bytes_sigma",
                         "bytes_total", "bpppc"])
        fcsv.flush()

    with open(os.path.join(DS, "splits", a.difficulty, a.split + ".csv")) as f:
        entries = [l.strip() for l in f if l.strip()]
    entries = entries[a.start:a.start + a.stride * a.num:a.stride]
    print("images to evaluate:", len(entries), flush=True)

    os.makedirs(a.workdir, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = a.gpu

    total_bits = 0
    total_px = 0
    n_done = 0
    for e in entries:
        tif = os.path.join(DS, "patches", e)
        name = os.path.basename(tif).replace("-SPECTRAL_IMAGE.TIF", "")

        if name in done:
            r = done[name]
            total_bits += int(r["bytes_total"]) * 8
            total_px += int(r["rows"]) * int(r["cols"]) * int(r["bands"])
            n_done += 1
            continue

        with rasterio.open(tif) as ds:
            img = ds.read(VALID)
        if a.clip:
            img = np.clip(img, 0, 10000)
        img = img.astype(np.uint16)
        nrows, ncols, nbands = img.shape[1], img.shape[2], img.shape[0]
        inp = os.path.join(a.workdir, name + ".bsq")
        with open(inp, "wb") as f:
            for b in range(nbands):
                f.write(np.ascontiguousarray(img[b]).astype("<u2").tobytes())

        prefix = os.path.join(outdir, name + "_d%d" % a.delta)
        cmd = [sys.executable, "compressor.py",
               "--model_file", cfg_model, "--config_file", cfg_file,
               "--input_image", inp, "--input_image_interleave", "BSQ",
               "--N_rows", str(nrows), "--N_cols", str(ncols), "--N_bands", str(nbands),
               "--side_info_file", prefix + "_side_info.h5",
               "--numerical_warning_file", prefix + "_numerical.bin",
               "--output_file", prefix + "_compressed.cmp",
               "--residuals_file", prefix + "_residuals.bsq",
               "--output_image", prefix + "_reconstructed.mat",
               "--entropy_encoder_binary", BIN_ENCODER,
               "--delta_quantization", str(a.delta), "--device", "cuda"]
        t_start = time.time()
        rc = subprocess.run(cmd, cwd=os.path.join(ROOT, "Code", "linerwkv"), env=env)
        if rc.returncode != 0:
            print("FAILED", name, flush=True)
            continue
        ratecmp = os.path.getsize(prefix + "_compressed.cmp")
        ratenum = os.path.getsize(prefix + "_numerical.bin")
        ratesidemu = os.path.getsize(prefix + "_side_info.h5_mu.cmp")
        ratesidesigma = os.path.getsize(prefix + "_side_info.h5_sigma.cmp")
        rate = ratecmp + ratenum + 64 * nbands + ratesidemu + ratesidesigma
        bits = rate * 8
        px = nrows * ncols * nbands
        bpppc = bits / px
        print("%s: %d bytes  bpppc=%.4f  (%.1fs)  [%d/%d]"
              % (name, rate, bpppc, time.time() - t_start, n_done + 1, len(entries)), flush=True)
        writer.writerow([name, nrows, ncols, nbands, ratecmp, ratenum,
                         ratesidemu, ratesidesigma, rate, "%.6f" % bpppc])
        fcsv.flush()
        total_bits += bits
        total_px += px
        n_done += 1

    fcsv.close()
    if total_px:
        print("AVERAGE bpppc over %d images (delta=%d, clip=%d, tag=%s): %.4f"
              % (n_done, a.delta, a.clip, tag, total_bits / total_px), flush=True)


if __name__ == "__main__":
    main()
