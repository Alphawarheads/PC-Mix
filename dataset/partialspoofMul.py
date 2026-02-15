import os
import torch
import random
import numpy as np
import soundfile as sf
import lightning as L

from utils import *
from dataset.base_dataset import BaseDataset
from torch.utils.data import DataLoader


class PartialSpoofDataModule(L.LightningDataModule):
    """
    DataModule 输出的 batch 结构要和 train_bam_multihead.py 对齐：

    (
      utt_id,
      mix_input, ref_speech, ref_env,              # 波形

      ori_label, boundary_label,                   # mix 分支帧级标签
      ori_label_length, boundary_length,           # mix 分支有效帧长

      sp_label, sp_boundary_label,                 # speech 组件帧级标签
      sp_label_length, sp_boundary_length,         # speech 组件有效帧长

      env_label, env_boundary_label,               # env 组件帧级标签
      env_label_length, env_boundary_length,       # env 组件有效帧长

      label_speech_utt, label_env_utt, label_mix_utt,  # utter-level 0/1 标签
    )
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_dataset('train')
            self.vlidate_dataset = self.get_dataset('dev')

        if stage == 'test' or stage is None:
            self.test_dataset = self.get_dataset('eval')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.vlidate_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers
        )

    def get_dataset(self, type):
        dataset = PartialSpoofDataset(
            samplerate=self.args.samplerate,
            resolution=self.args.resolution,
            root=getattr(self.args, f'{type}_root'),   # mix 音频根目录
            input_type=type,
            input_maxlength=self.args.input_maxlength,
            input_minlength=self.args.input_minlength,
            input_query='*.wav',
            input_load_fn=None,
            label_root=self.args.label_root,
            label_load_fn=None,
            label_maxlength=self.args.label_maxlength,
            pad_mode=self.args.pad_mode if type == 'train' else None,
            add_label=True,

            # 新增：ref 语音 / 背景根目录
            ref_speech_root=getattr(self.args, 'ref_speech_root', None),
            ref_env_root=getattr(self.args, 'ref_env_root', None),
        )
        return dataset


class PartialSpoofDataset(BaseDataset):
    """
    继承 BaseDataset：
      - BaseDataset 的 __getitem__ 通常会先用 default_input_load_fn + default_label_load_fn
        得到 (utt_id, mix_input, ori_label, ori_label_length)
      - 然后调用 add_other_label，再调用 pad

    我们在 add_other_label 里补齐：
      - ref_speech / ref_env 波形
      - mix/speech/env 的 boundary
      - speech/env 的帧级标签
      - utter-level 标签
    """

    def __init__(self, *args, ref_speech_root=None, ref_env_root=None, **kwargs):
        self.ref_speech_root = ref_speech_root
        self.ref_env_root = ref_env_root
        super().__init__(*args, **kwargs)

        # ====== 预加载组件帧级标签 & utter 级标签 ======
        # 1) speech 组件帧级标签：{utt_id: np.array(T_sp,)}
        sp_label_file = os.path.join(
            self.label_root, f'{self.input_type}_speech_seglab_{self.resolution}.npy'
        )
        if os.path.exists(sp_label_file):
            sp_labels = np.load(sp_label_file, allow_pickle=True).item()
            self.sp_labels = {k: v.astype(int) for k, v in sp_labels.items()}
        else:
            self.sp_labels = {}

        # 2) env 组件帧级标签：{utt_id: np.array(T_env,)}
        env_label_file = os.path.join(
            self.label_root, f'{self.input_type}_env_seglab_{self.resolution}.npy'
        )
        if os.path.exists(env_label_file):
            env_labels = np.load(env_label_file, allow_pickle=True).item()
            self.env_labels = {k: v.astype(int) for k, v in env_labels.items()}
        else:
            self.env_labels = {}

        # 3) utter-level 标签：{utt_id: {"speech":0/1, "env":0/1, "mix":0/1}} 或 tuple
        utt_label_file = os.path.join(self.label_root, f'{self.input_type}_utt_labels.npy')
        if os.path.exists(utt_label_file):
            self.utt_labels = np.load(utt_label_file, allow_pickle=True).item()
        else:
            self.utt_labels = {}

    def default_input_load_fn(self, path):
        audio, sr = sf.read(path)
        # 转成 torch.Tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.ndim > 1:
            audio = audio.mean(-1)  # 单通道
        return audio

    def default_label_load_fn(self):
        """
        这里沿用原来的 mix 分支帧级标签：{utt_id: np.array(T_mix,)}
        """
        label_file = os.path.join(
            self.label_root, f'{self.input_type}_seglab_{self.resolution}.npy'
        )
        labels = np.load(label_file, allow_pickle=True).item()
        labels = {k: v.astype(int) for k, v in labels.items()}
        return labels

    def _load_ref_wav(self, utt_id, root):
        """
        参考语音 / 背景的 wav loader。
        默认假设文件名是 {utt_id}.wav，你可以根据自己情况改命名规则。
        """
        if root is None:
            # 没提供 ref_root，就用全 0 替代（不会崩，但分离 MSE 没意义）
            return torch.zeros(1)
        wav_path = os.path.join(root, f'{utt_id}.wav')
        audio, sr = sf.read(wav_path)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.ndim > 1:
            audio = audio.mean(-1)
        return audio

    def _load_boundary(self, utt_id, kind='mix'):
        """
        kind: "mix" / "speech" / "env"
        对应：

          boundary_{res}_labels/{type}/{utt_id}_boundary.npy
          boundary_speech_{res}_labels/{type}/{utt_id}_boundary.npy
          boundary_env_{res}_labels/{type}/{utt_id}_boundary.npy

        你如果实际目录名字不一样，改这里就行。
        """
        if kind == 'mix':
            root = os.path.join(
                self.label_root,
                f'boundary_{self.resolution}_labels',
                self.input_type
            )
        elif kind == 'speech':
            root = os.path.join(
                self.label_root,
                f'boundary_speech_{self.resolution}_labels',
                self.input_type
            )
        elif kind == 'env':
            root = os.path.join(
                self.label_root,
                f'boundary_env_{self.resolution}_labels',
                self.input_type
            )
        else:
            raise ValueError(f'Unknown boundary kind: {kind}')

        path = os.path.join(root, f'{utt_id}_boundary.npy')
        boundary_label = np.load(path).astype(np.float32)
        return torch.from_numpy(boundary_label)

    def add_other_label(self, items):
        """
        BaseDataset 默认先返回：
          items = (utt_id, mix_input, ori_label, ori_label_length)

        我们在这里扩展成训练脚本需要的那一大坨：
          (
            utt_id,
            mix_input, ref_speech, ref_env,
            ori_label, boundary_label,
            ori_label_length, boundary_length,
            sp_label, sp_boundary_label,
            sp_label_length, sp_boundary_length,
            env_label, env_boundary_label,
            env_label_length, env_boundary_length,
            label_speech_utt, label_env_utt, label_mix_utt,
          )
        """
        utt_id, mix_input, ori_label, ori_label_length = items

        # ===== 1) 参考语音 / 背景 wav =====
        ref_speech = self._load_ref_wav(utt_id, self.ref_speech_root)
        ref_env = self._load_ref_wav(utt_id, self.ref_env_root)

        # ===== 2) mix 分支 boundary =====
        boundary_label = self._load_boundary(utt_id, kind='mix')
        boundary_length = len(boundary_label)

        # ===== 3) speech 组件帧级标签 + boundary =====
        if utt_id in self.sp_labels:
            sp_label_np = self.sp_labels[utt_id]
        else:
            # 没有就用全 0 填（你可以改成报错）
            sp_label_np = np.zeros_like(ori_label.numpy())
        sp_label = torch.from_numpy(sp_label_np).long()
        sp_label_length = len(sp_label)

        sp_boundary_label = self._load_boundary(utt_id, kind='speech')
        sp_boundary_length = len(sp_boundary_label)

        # ===== 4) env 组件帧级标签 + boundary =====
        if utt_id in self.env_labels:
            env_label_np = self.env_labels[utt_id]
        else:
            env_label_np = np.zeros_like(ori_label.numpy())
        env_label = torch.from_numpy(env_label_np).long()
        env_label_length = len(env_label)

        env_boundary_label = self._load_boundary(utt_id, kind='env')
        env_boundary_length = len(env_boundary_label)

        # ===== 5) utter-level 标签 =====
        if utt_id in self.utt_labels:
            utt_info = self.utt_labels[utt_id]
            # 支持 dict 或 tuple/list
            if isinstance(utt_info, dict):
                label_speech_utt = int(utt_info.get('speech', 0))
                label_env_utt = int(utt_info.get('env', 0))
                label_mix_utt = int(utt_info.get('mix', 0))
            else:
                # 假设 (speech, env, mix)
                label_speech_utt = int(utt_info[0])
                label_env_utt = int(utt_info[1])
                label_mix_utt = int(utt_info[2])
        else:
            label_speech_utt = 0
            label_env_utt = 0
            label_mix_utt = 0

        label_speech_utt = torch.tensor(label_speech_utt, dtype=torch.long)
        label_env_utt = torch.tensor(label_env_utt, dtype=torch.long)
        label_mix_utt = torch.tensor(label_mix_utt, dtype=torch.long)

        new_items = (
            utt_id,
            mix_input, ref_speech, ref_env,
            ori_label, boundary_label,
            ori_label_length, boundary_length,
            sp_label, sp_boundary_label,
            sp_label_length, sp_boundary_length,
            env_label, env_boundary_label,
            env_label_length, env_boundary_length,
            label_speech_utt, label_env_utt, label_mix_utt,
        )
        return new_items

    def pad(self, items):
        """
        训练阶段 pad/crop 到统一长度（按 mix 分支的帧数和 label_maxlength 来做），
        确保：

          - 三路帧标签长度一致
          - 三路 boundary 长度一致
          - 三路 wav 长度 = 帧数 * scale

        NOTE: 这里假设三路标签在同一时间网格上（常见做法），如果你自己的长度不一样，
              要自己根据实际情况改这一块。
        """
        if self.pad_mode != 'label':
            return items

        (utt_id,
         mix_input, ref_speech, ref_env,
         ori_label, boundary_label,
         ori_label_length, boundary_length,
         sp_label, sp_boundary_label,
         sp_label_length, sp_boundary_length,
         env_label, env_boundary_label,
         env_label_length, env_boundary_length,
         label_speech_utt, label_env_utt, label_mix_utt) = items

        # 基础参数
        scale = int(self.samplerate * self.resolution)
        # 以 mix 的帧数为主
        L = int(ori_label_length)
        target_L = L

        # 如果有 label_maxlength，就限制在 [0, label_maxlength]
        if self.label_maxlength is not None:
            if target_L < self.label_maxlength:
                # ====== pad 到 label_maxlength ======
                pad_frames = self.label_maxlength - target_L
                target_L = self.label_maxlength

                def pad_1d(x, value=0):
                    return torch.nn.functional.pad(x, (0, pad_frames), mode='constant', value=value)

                ori_label = pad_1d(ori_label, 0)
                boundary_label = pad_1d(boundary_label, 0.0)

                sp_label = pad_1d(sp_label, 0)
                sp_boundary_label = pad_1d(sp_boundary_label, 0.0)

                env_label = pad_1d(env_label, 0)
                env_boundary_label = pad_1d(env_boundary_label, 0.0)

                def pad_wav(x):
                    wav_pad = target_L * scale - x.size(0)
                    if wav_pad > 0:
                        x = torch.nn.functional.pad(x, (0, wav_pad), mode='constant', value=0.0)
                    return x

                mix_input = pad_wav(mix_input)
                ref_speech = pad_wav(ref_speech)
                ref_env = pad_wav(ref_env)

            elif target_L > self.label_maxlength:
                # ====== 随机裁剪到 label_maxlength ======
                target_L = self.label_maxlength
                startp = random.randint(0, L - target_L)
                endp = startp + target_L

                ori_label = ori_label[startp:endp]
                boundary_label = boundary_label[startp:endp]

                sp_label = sp_label[startp:endp]
                sp_boundary_label = sp_boundary_label[startp:endp]

                env_label = env_label[startp:endp]
                env_boundary_label = env_boundary_label[startp:endp]

                def crop_wav(x):
                    s = startp * scale
                    e = endp * scale
                    x = x[s:e]
                    return x

                mix_input = crop_wav(mix_input)
                ref_speech = crop_wav(ref_speech)
                ref_env = crop_wav(ref_env)

        # 注意：这里沿用原代码风格，不去改 *_length（它们还是原始长度），
        # get_src_mask 一般会用 label 的 shape 限制 max_len，所以不会越界。
        new_items = (
            utt_id,
            mix_input, ref_speech, ref_env,
            ori_label, boundary_label,
            ori_label_length, boundary_length,
            sp_label, sp_boundary_label,
            sp_label_length, sp_boundary_length,
            env_label, env_boundary_label,
            env_label_length, env_boundary_length,
            label_speech_utt, label_env_utt, label_mix_utt,
        )
        return new_items


if __name__ == '__main__':
    print('define of partialspoof dataset')
