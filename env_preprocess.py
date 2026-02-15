import os
from tqdm import tqdm

import glob
import librosa
import soundfile
import numpy as np
import shutil


AUDIO_EXTS = (".wav", ".flac")


def list_audio_files(root, subdir):
    """递归列出 root/subdir 下面所有 wav/flac 文件"""
    audio_files = []
    base = os.path.join(root, subdir)
    for ext in AUDIO_EXTS:
        audio_files.extend(glob.glob(os.path.join(base, f"**/*{ext}"), recursive=True))
    return sorted(audio_files)


def preprocess(root, cache_root, subdir, samplerate):
    """
    从 root/subdir 读取原始音频，重采样到 samplerate，
    存到 cache_root/raw/subdir 下面（统一为 .wav）
    """
    raw_list = list_audio_files(root, subdir)
    save_dir = os.path.join(cache_root, "raw", subdir)
    os.makedirs(save_dir, exist_ok=True)

    for path in tqdm(raw_list, desc=f"Resampling {subdir}..."):
        data, sr = librosa.load(path, sr=samplerate)
        name = os.path.splitext(os.path.basename(path))[0]
        sp = os.path.join(save_dir, f"{name}.wav")
        soundfile.write(sp, data, samplerate)


def get_boundary_labels(root, subdir, cache_root,
                        labels_filename="partial_labels_20ms.npy",
                        resolution_tag="20ms"):
    """
    适配你的 partial_eval 结构：
      root/subdir                 -> 音频目录（如 partial_env）
      root/labels_filename        -> seglab .npy（如 partial_labels_20ms.npy）

    生成：
      cache_root/labels_filename (备份)
      cache_root/boundary_{resolution_tag}_labels/*.npy
    """
    data_root = os.path.join(root, subdir)
    labels_path = os.path.join(root, labels_filename)

    # 读取 seglab 字典
    labels_dict = np.load(labels_path, allow_pickle=True).item()
    utt_list = list_audio_files(root, subdir)

    # 备份 seglab 到 cache_root
    os.makedirs(cache_root, exist_ok=True)
    shutil.copy(labels_path, os.path.join(cache_root, labels_filename))

    # 保存边界标签的目录
    save_dir = os.path.join(cache_root, f"boundary_{resolution_tag}_labels")
    os.makedirs(save_dir, exist_ok=True)

    all_count = 0
    boundary_count = 0

    for utt in tqdm(utt_list, desc=f"Get boundary label for {subdir}..."):
        name = os.path.splitext(os.path.basename(utt))[0]

        if name not in labels_dict:
            # 如果 label 里没有这个 key，就跳过
            # 也可以选择 raise，这里先稳妥一点
            # print(f"[WARN] {name} not in labels dict, skip.")
            continue

        # 兼容 list / ndarray 两种格式
        utt_label = labels_dict[name]
        if not isinstance(utt_label, np.ndarray):
            utt_label = np.array(utt_label, dtype=np.int32)
        else:
            utt_label = utt_label.astype(np.int32)

        all_count += len(utt_label)

        # 找边界：和官方脚本一样，label 变化的地方前一帧标记为 1
        pos = []
        last = utt_label[0]
        for i, label in enumerate(utt_label):
            if label != last:
                splice_index = i if label == 0 else i - 1
                pos.append(splice_index)
                last = label

        pos = list(set(pos))
        boundary_count += len(pos)

        boundary_label = np.zeros_like(utt_label, dtype=np.float32)
        boundary_label[pos] = 1.0

        np.save(os.path.join(save_dir, f"{name}_boundary.npy"), boundary_label)

    if boundary_count > 0:
        print(f"pos_weight: {(all_count - boundary_count) / (boundary_count)}")
    else:
        print("No boundary found, please check labels.")


if __name__ == '__main__':
    # === 根据你现在的结构设置 ===
    partial_eval_root = "/DATA1/Audiodata/AntiSpoofingData/partial_eval"
    audio_subdir = "partial_env"              # 图片里的这个文件夹
    labels_filename = "partial_labels_20ms.npy"
    data_cache_path = "./data_partial_eval"   # 随便放一个输出目录
    samplerate = 16000

    # 1) 重采样音频（可选，如果已经是 16k 也可以注释掉）
    preprocess(partial_eval_root, data_cache_path, audio_subdir, samplerate)

    # 2) 生成边界标签
    get_boundary_labels(
        partial_eval_root,
        audio_subdir,
        data_cache_path,
        labels_filename=labels_filename,
        resolution_tag="20ms"
    )
