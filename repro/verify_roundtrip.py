"""Verify LineRWKV lossless round-trip for one HySpecNet-11k test image.

Checks, in order:
  1. compressor .mat  == clipped input image            (lossless on the encoder side)
  2. decompressor .mat == compressor .mat               (real end-to-end decodability)
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import rasterio
import scipy.io as sio

INVALID = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
           160, 161, 162, 163, 164, 165, 166]
VALID = [c + 1 for c in range(224) if c not in INVALID]

# NOTE: lab-local absolute paths were replaced by placeholders for this public copy.
# Point them at your own checkout / dataset root before running.
ROOT = "<REPO_ROOT>"
DS = "<DATASET_ROOT>"
BIN_ENCODER = os.path.join(ROOT, "Code", "compressor")
BIN_DECODER = os.path.join(ROOT, "Code", "decompressor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="xs_1500")
    ap.add_argument("--tag", default="")
    ap.add_argument("--difficulty", default="hard")
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--delta", type=int, default=0)
    ap.add_argument("--gpu", default="4")
    a = ap.parse_args()

    tag = a.tag or a.config
    outdir = os.path.join(ROOT, "Results", "linerwkv", tag)
    cfg_file = os.path.join(ROOT, "log_dir", "linerwkv", a.config, "config.txt")
    model_file = os.path.join(ROOT, "Results", "linerwkv", a.config, "rwkv-final.pth")

    with open(os.path.join(DS, "splits", a.difficulty, a.split + ".csv")) as f:
        entry = [l.strip() for l in f if l.strip()][a.index]
    name = os.path.basename(entry).replace("-SPECTRAL_IMAGE.TIF", "")
    workdir = "<WORK_DIR>"
    bsq = os.path.join(workdir, name + ".bsq")
    prefix = os.path.join(outdir, name + "_d%d" % a.delta)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = a.gpu
    py = sys.executable

    cmd = [py, "compressor.py", "--model_file", model_file, "--config_file", cfg_file,
           "--input_image", bsq, "--input_image_interleave", "BSQ",
           "--N_rows", "128", "--N_cols", "128", "--N_bands", "202",
           "--side_info_file", prefix + "_side_info.h5",
           "--numerical_warning_file", prefix + "_numerical.bin",
           "--output_file", prefix + "_compressed.cmp",
           "--residuals_file", prefix + "_residuals.bsq",
           "--output_image", prefix + "_reconstructed.mat",
           "--entropy_encoder_binary", BIN_ENCODER,
           "--delta_quantization", str(a.delta), "--device", "cuda"]
    label = "[1/2] skip re-compress (use existing)"
    if not os.path.exists(prefix + "_compressed.cmp"):
        label = "[1/2] compress"
        rc = subprocess.run(cmd, cwd=os.path.join(ROOT, "Code", "linerwkv"), env=env)
        assert rc.returncode == 0, "compressor failed"
    print(label, flush=True)

    enc = sio.loadmat(prefix + "_reconstructed.mat")["data_rec"]
    with rasterio.open(os.path.join(DS, "patches", entry)) as ds:
        ref = np.clip(ds.read(VALID), 0, 10000).astype(np.uint16)
    ref = np.transpose(ref, (1, 2, 0))  # (H,W,C)
    enc_r = np.rint(enc).astype(np.int64)
    d1 = np.abs(enc_r - ref.astype(np.int64))
    print("encoder-side: max|diff| = %d, #diff = %d / %d, identical = %s"
          % (d1.max(), int((d1 > 0).sum()), d1.size, bool((d1 == 0).all())), flush=True)

    tmp_mat = prefix + "_decoded_check.mat"
    cmd2 = [py, "decompressor.py", "--model_file", model_file, "--config_file", cfg_file,
            "--output_image", tmp_mat,
            "--N_rows", "128", "--N_cols", "128", "--N_bands", "202",
            "--compressed_file", prefix + "_compressed.cmp",
            "--side_info_file", prefix + "_side_info.h5",
            "--residuals_file", prefix + "_decoded_tmp.bsq",
            "--numerical_warning_file", prefix + "_numerical.bin",
            "--delta_quantization", str(a.delta), "--device", "cuda",
            "--entropy_decoder_binary", BIN_DECODER]
    print("[2/2] decompress", flush=True)
    rc = subprocess.run(cmd2, cwd=os.path.join(ROOT, "Code", "linerwkv"), env=env)
    assert rc.returncode == 0, "decompressor failed"
    dec = sio.loadmat(tmp_mat)["data_rec"]
    d2 = np.abs(np.rint(dec).astype(np.int64) - np.rint(enc).astype(np.int64))
    print("decoder-side: max|diff| = %d, #diff = %d / %d, identical = %s"
          % (d2.max(), int((d2 > 0).sum()), d2.size, bool((d2 == 0).all())), flush=True)
    ok = (d1 == 0).all() and (d2 == 0).all()
    print("ROUND-TRIP LOSSLESS:", "PASS" if ok else "FAIL", flush=True)
    print("image:", name)


if __name__ == "__main__":
    main()
