#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dummy data generator for BAM multi-head pipeline.

Key semantic (per your requirement):
  - frame seg label: 1 = tampered/spoof region, 0 = clean/original region
  - boundary label: 1 = change point of seg label (0<->1), else 0
  - "original" utterance (same-recording): all seg labels are 0 and boundary all 0

Files produced under out_root:

  {out_root}/{split}/wav/{utt_id}.wav                         (mix)
  {out_root}/ref_speech/{utt_id}.wav
  {out_root}/ref_env/{utt_id}.wav

  {out_root}/labels/{split}_seglab_{resolution}.npy           (dict: utt_id -> (T,) int64)
  {out_root}/labels/{split}_speech_seglab_{resolution}.npy
  {out_root}/labels/{split}_env_seglab_{resolution}.npy

  {out_root}/labels/boundary_{resolution}_labels/{split}/{utt_id}_boundary.npy
  {out_root}/labels/boundary_speech_{resolution}_labels/{split}/{utt_id}_boundary.npy
  {out_root}/labels/boundary_env_{resolution}_labels/{split}/{utt_id}_boundary.npy

  {out_root}/labels/{split}_utt_labels.npy
    dict: utt_id -> {"speech": int, "env": int, "mix": int}

IMPORTANT:
  - By default, utt_labels["mix"] is "mix_tampered": 1 = non-original/tampered, 0 = original.
    This matches "label=1 means tampered" convention.
  - If your DataModule expects "mix=1 is original", set --mix_label_mode orig.
"""

import os
import argparse
import numpy as np
import soundfile as sf


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_wav(path: str, n_samples: int, sr: int, rms: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples).astype(np.float32)
    # normalize to target rms
    cur = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    x = x / cur * float(rms)
    sf.write(path, x, sr)


def make_seglab(T: int, p_tamper: float, seed: int = 0):
    """
    生成 0/1 seg label: 1 = tampered/spoof, 0 = clean/original
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(T) < p_tamper).astype(np.int64)
    return y


def seglab_to_boundary(y: np.ndarray):
    """
    y: (T,) 0/1, 1=tampered
    boundary: (T,) float32
    标记发生切换的位置（0->1 或 1->0）
    """
    T = int(y.shape[0])
    b = np.zeros(T, dtype=np.float32)
    if T <= 1:
        return b
    b[1:] = (y[1:] != y[:-1]).astype(np.float32)
    return b


def gen_split(
    out_root: str,
    split: str,
    num_utts: int,
    sr: int,
    resolution: float,
    T: int,
    seed: int,
    p_original: float = 0.25,
    mix_label_mode: str = "tamper",  # "tamper" or "orig"
    p_tamper_mix: float = 0.35,
    p_tamper_sp: float = 0.25,
    p_tamper_env: float = 0.45,
):
    """
    生成一个 split（train/dev/eval）。

    mix_label_mode:
      - "tamper": utt_labels["mix"]=1 means non-original/tampered, 0 means original (RECOMMENDED)
      - "orig":   utt_labels["mix"]=1 means original, 0 means non-original
    """
    mix_root = os.path.join(out_root, split, "wav")
    ref_speech_root = os.path.join(out_root, "ref_speech")
    ref_env_root = os.path.join(out_root, "ref_env")
    label_root = os.path.join(out_root, "labels")

    ensure_dir(mix_root)
    ensure_dir(ref_speech_root)
    ensure_dir(ref_env_root)
    ensure_dir(label_root)

    # boundary dirs
    b_mix_dir = os.path.join(label_root, f"boundary_{resolution}_labels", split)
    b_sp_dir = os.path.join(label_root, f"boundary_speech_{resolution}_labels", split)
    b_env_dir = os.path.join(label_root, f"boundary_env_{resolution}_labels", split)
    ensure_dir(b_mix_dir)
    ensure_dir(b_sp_dir)
    ensure_dir(b_env_dir)

    # dict labels
    mix_seglab = {}
    sp_seglab = {}
    env_seglab = {}
    utt_labels = {}

    scale = int(sr * resolution)
    wav_len = T * scale  # 保证 wav 长度与 label 对齐

    # 固定比例 original：每 orig_interval 条做一条 original
    orig_interval = max(1, int(round(1.0 / max(p_original, 1e-8))))

    for i in range(num_utts):
        utt_id = f"{split}_utt_{i:05d}"
        s = seed + i * 13

        # 是否设置为 original 样本
        is_original = (i % orig_interval == 0)

        # 1) wav (random noise for dummy)
        make_wav(os.path.join(mix_root, f"{utt_id}.wav"), wav_len, sr, seed=s)
        make_wav(os.path.join(ref_speech_root, f"{utt_id}.wav"), wav_len, sr, seed=s + 1)
        make_wav(os.path.join(ref_env_root, f"{utt_id}.wav"), wav_len, sr, seed=s + 2)

        # 2) seglab（0/1; 1=tampered）
        if is_original:
            y_mix = np.zeros(T, dtype=np.int64)
            y_sp = np.zeros(T, dtype=np.int64)
            y_env = np.zeros(T, dtype=np.int64)
        else:
            y_mix = make_seglab(T, p_tamper=p_tamper_mix, seed=s + 3)
            y_sp = make_seglab(T, p_tamper=p_tamper_sp, seed=s + 4)
            y_env = make_seglab(T, p_tamper=p_tamper_env, seed=s + 5)

            # avoid degenerate "all 0" for non-original (optional but helps training sanity)
            if np.all(y_mix == 0):
                y_mix[np.random.default_rng(s + 30).integers(0, T)] = 1
            if np.all(y_sp == 0):
                y_sp[np.random.default_rng(s + 31).integers(0, T)] = 1
            if np.all(y_env == 0):
                y_env[np.random.default_rng(s + 32).integers(0, T)] = 1

        mix_seglab[utt_id] = y_mix
        sp_seglab[utt_id] = y_sp
        env_seglab[utt_id] = y_env

        # 3) boundary（float32）从 seglab 推导，确保语义一致
        b_mix = seglab_to_boundary(y_mix)
        b_sp = seglab_to_boundary(y_sp)
        b_env = seglab_to_boundary(y_env)

        np.save(os.path.join(b_mix_dir, f"{utt_id}_boundary.npy"), b_mix.astype(np.float32))
        np.save(os.path.join(b_sp_dir, f"{utt_id}_boundary.npy"), b_sp.astype(np.float32))
        np.save(os.path.join(b_env_dir, f"{utt_id}_boundary.npy"), b_env.astype(np.float32))

        # 4) utt-level labels (keep "1 means tampered" for speech/env)
        sp_utt = int(np.any(y_sp == 1))      # 1=has tamper in speech
        env_utt = int(np.any(y_env == 1))    # 1=has tamper in env

        if mix_label_mode == "orig":
            mix_lbl = int(is_original)       # 1=original, 0=non-original
        else:
            mix_lbl = int(not is_original)   # 1=tampered/non-original, 0=original

        utt_labels[utt_id] = {
            "speech": sp_utt,
            "env": env_utt,
            "mix": mix_lbl,
        }

    # 保存 dict npy（allow_pickle=True 才能 item()）
    np.save(os.path.join(label_root, f"{split}_seglab_{resolution}.npy"), mix_seglab)
    np.save(os.path.join(label_root, f"{split}_speech_seglab_{resolution}.npy"), sp_seglab)
    np.save(os.path.join(label_root, f"{split}_env_seglab_{resolution}.npy"), env_seglab)
    np.save(os.path.join(label_root, f"{split}_utt_labels.npy"), utt_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="./fake_partialspoof_data")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--resolution", type=float, default=0.16)
    ap.add_argument("--T", type=int, default=20, help="frame length (<= label_maxlength 推荐)")
    ap.add_argument("--num_train", type=int, default=32)
    ap.add_argument("--num_dev", type=int, default=8)
    ap.add_argument("--num_eval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--p_original", type=float, default=0.25,
                    help="ratio of original utterances")
    ap.add_argument("--mix_label_mode", type=str, default="tamper", choices=["tamper", "orig"],
                    help='utter-level mix label meaning: "tamper"=1 non-original, "orig"=1 original')

    ap.add_argument("--p_tamper_mix", type=float, default=0.35)
    ap.add_argument("--p_tamper_sp", type=float, default=0.25)
    ap.add_argument("--p_tamper_env", type=float, default=0.45)

    args = ap.parse_args()

    # 生成三个 split
    gen_split(args.out_root, "train", args.num_train,
              args.sr, args.resolution, args.T, args.seed,
              p_original=args.p_original,
              mix_label_mode=args.mix_label_mode,
              p_tamper_mix=args.p_tamper_mix,
              p_tamper_sp=args.p_tamper_sp,
              p_tamper_env=args.p_tamper_env)

    gen_split(args.out_root, "dev", args.num_dev,
              args.sr, args.resolution, args.T, args.seed + 999,
              p_original=args.p_original,
              mix_label_mode=args.mix_label_mode,
              p_tamper_mix=args.p_tamper_mix,
              p_tamper_sp=args.p_tamper_sp,
              p_tamper_env=args.p_tamper_env)

    gen_split(args.out_root, "eval", args.num_eval,
              args.sr, args.resolution, args.T, args.seed + 1999,
              p_original=args.p_original,
              mix_label_mode=args.mix_label_mode,
              p_tamper_mix=args.p_tamper_mix,
              p_tamper_sp=args.p_tamper_sp,
              p_tamper_env=args.p_tamper_env)

    print("\nDone.")
    print("Mix wav roots:")
    print(f"  train_root = {os.path.abspath(os.path.join(args.out_root, 'train', 'wav'))}")
    print(f"  dev_root   = {os.path.abspath(os.path.join(args.out_root, 'dev', 'wav'))}")
    print(f"  eval_root  = {os.path.abspath(os.path.join(args.out_root, 'eval', 'wav'))}")
    print("Ref roots:")
    print(f"  ref_speech_root = {os.path.abspath(os.path.join(args.out_root, 'ref_speech'))}")
    print(f"  ref_env_root    = {os.path.abspath(os.path.join(args.out_root, 'ref_env'))}")
    print("Label root:")
    print(f"  label_root = {os.path.abspath(os.path.join(args.out_root, 'labels'))}")
    print("\nQuick check filenames:")
    print(f"  labels/train_seglab_{args.resolution}.npy")
    print(f"  labels/train_utt_labels.npy")
    print(f"  labels/boundary_{args.resolution}_labels/train/<utt>_boundary.npy")
    print("\nNOTE:")
    print(f'  seg label: 1=tampered, 0=clean')
    print(f'  mix utt label mode: {args.mix_label_mode} (mix=1 means {"non-original/tampered" if args.mix_label_mode=="tamper" else "original"})')


if __name__ == "__main__":
    main()
