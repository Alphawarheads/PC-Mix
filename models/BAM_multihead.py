import torch
import torch.nn as nn

from models.Unet_mask import UNetSTFTComplexRefine
from models.bam import BAM


class BAMMultiHeadModel(nn.Module):
    """
    UNet 分离 + 3 个 BAM 头的 multi-head 模型：

    输入：
      mix_input   : (B, T)   混合波形
      ref_speech  : (B, T_s) 参考 “纯语音” 波形（来自你的 bg_partial / ori 预处理）
      ref_env     : (B, T_e) 参考 “纯背景” 波形

    输出（按 train_bam_multihead.py 里的约定顺序）：

      frame_mix        : (B, Tm, 2)   mix 上的帧级 spoof logits
      boundary_mix     : (B, Tm)      mix 上的 boundary 概率

      frame_sp_hat     : (B, Ts, 2)   分离 speech_hat 上的帧级 logits
      boundary_sp_hat  : (B, Ts)      分离 speech_hat 上的 boundary

      frame_env_hat    : (B, Te, 2)   分离 env_hat 上的帧级 logits
      boundary_env_hat : (B, Te)      分离 env_hat 上的 boundary

      speech_hat       : (B, T)       UNet 分离出来的语音
      env_hat          : (B, T)       UNet 分离出来的背景

      logits_mix       : (B, 2)       utter-level，判断 mix 是否为 partial spoof（label_mix）
      logits_sp_ref    : (B, 2)       utter-level，参考纯语音是否 spoof（label_speech）
      logits_env_ref   : (B, 2)       utter-level，参考纯背景是否 spoof（label_env）
      logits_sp_hat    : (B, 2)       utter-level，分离 speech_hat 的 spoof 判别      logits_env_hat   : (B, 2)       utter-level，分离 env_hat 的 spoof 判别

    说明：
      - BAM 本身还是原来的帧级 detector，不动。
      - utter-level logits 这里用 “时间平均 + 一个小线性层” 实现，简单但够用。
    """

    def __init__(self, args, config, device="cuda"):
        super().__init__()
        self.device = device

        # 1) UNet 分离器（共享）
        self.spar = UNetSTFTComplexRefine()

        # 2) 三个独立 BAM（参数不共享）
        #    如果你想省显存，也可以让它们共享，但先保持独立更安全
        self.bam_all    = BAM(args, config)  # 混合音频上的 detector
        self.bam_speech = BAM(args, config)  # 语音分量上的 detector
        self.bam_env    = BAM(args, config)  # 背景分量上的 detector

        # 3) utter-level heads：时间平均后的 2 维特征 -> 2 类 logits
        #    （这里假定 BAM 的最后一维就是 2 维 [bona, spoof] 的 logits）
        self.fc_mix      = nn.Linear(2, 2)
        self.fc_sp_ref   = nn.Linear(2, 2)
        self.fc_env_ref  = nn.Linear(2, 2)
        self.fc_sp_hat   = nn.Linear(2, 2)
        self.fc_env_hat  = nn.Linear(2, 2)

    def _utt_head(self, frame_logits, fc_layer):
        """
        frame_logits: (B, T, 2)
        返回 utter-level logits: (B, 2)

        这里用简单的时间平均做 pooling，你后面如果想改成 attention pooling 也可以。
        """
        # (B, T, 2) -> (B, 2)
        feat = frame_logits.mean(dim=1)
        return fc_layer(feat)

    def forward(self, mix_input, ref_speech, ref_env):
        """
        mix_input : (B, T)
        ref_speech: (B, T_s)
        ref_env   : (B, T_e)
        """

        # ===== 1) UNet 分离 =====
        # 输出和 mix_input 对齐的 speech_hat / env_hat
        speech_hat, env_hat = self.spar(mix_input)  # 假定 UNet 已经自己做了 padding / 裁剪

        # ===== 2) 三个 BAM：mix / speech_hat / env_hat =====
        # BAM 返回：frame_logits (B, T_frame, 2), boundary (B, T_frame)
        frame_mix,     boundary_mix     = self.bam_all(mix_input)
        frame_sp_hat,  boundary_sp_hat  = self.bam_speech(speech_hat)
        frame_env_hat, boundary_env_hat = self.bam_env(env_hat)

        # ===== 3) 参考真值分量：ref_speech / ref_env 也走一遍 BAM_speech / BAM_env =====
        frame_sp_ref,  _ = self.bam_speech(ref_speech)
        frame_env_ref, _ = self.bam_env(ref_env)

        # ===== 4) utter-level logits：时间平均 + 线性层 =====
        logits_mix      = self._utt_head(frame_mix,     self.fc_mix)
        logits_sp_ref   = self._utt_head(frame_sp_ref,  self.fc_sp_ref)
        logits_env_ref  = self._utt_head(frame_env_ref, self.fc_env_ref)
        logits_sp_hat   = self._utt_head(frame_sp_hat,  self.fc_sp_hat)
        logits_env_hat  = self._utt_head(frame_env_hat, self.fc_env_hat)

        # ===== 5) 按 train_bam_multihead.py 里约定的顺序返回 =====
        return (
            frame_mix, boundary_mix,
            frame_sp_hat, boundary_sp_hat,
            frame_env_hat, boundary_env_hat,
            speech_hat, env_hat,
            logits_mix, logits_sp_ref, logits_env_ref,
            logits_sp_hat, logits_env_hat,
        )
