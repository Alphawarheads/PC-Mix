import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskCrossEnrtopyLoss(nn.Module):
    """
    多类 / 二类 CrossEntropy，支持按时间维度的 mask（比如去掉 padding frame）
    pred: (B, C, T)  or (B, T, C) 但在这里我们会先转成 (B, C, T)
    target: (B, T)
    mask: (B, T) 取值 0/1
    """
    def __init__(self, weight=None):
        super(MaskCrossEnrtopyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, pred, target, mask=None):
        """
        pred: (B, C, T)
        target: (B, T)
        mask: (B, T) 或 None
        """
        B, C, T = pred.size()

        # IMPORTANT: pred 可能来自 transpose/permute，通常是 non-contiguous，
        # 用 reshape 替代 view 以避免 RuntimeError
        pred = pred.reshape(B * T, C)
        target = target.reshape(B * T)

        loss = self.ce(pred, target)  # (B*T,)

        if mask is not None:
            mask = mask.reshape(B * T)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss


class MaskBCELoss(nn.Module):
    """
    二分类 BCE，支持 mask；一般不做 hard negative 挖掘。
    """
    def __init__(self):
        super(MaskBCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target, mask=None):
        """
        pred: (B, T) or (B, 1, T) 等，内部会展平
        target: (B, T)
        mask: (B, T) or None
        """
        pred = pred.reshape_as(target)
        loss = self.bce(pred, target)

        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss


class BalanceBCELoss(nn.Module):
    """
    用于极度类别不平衡的 0/1 任务（比如 boundary）:
      - 对所有 positive frame 都算 loss
      - negative frame 里只取 top-k 最大的那些（hard negatives）
    """
    def __init__(self, negative_ratio=5.0, eps=1e-8):
        super(BalanceBCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, target, mask=None):
        """
        pred: (B, T)
        target: (B, T), 0/1
        mask: (B, T)，0/1，用于去掉 padding frame（没有就传全 1）
        """
        pred = pred.reshape_as(target)

        loss = F.binary_cross_entropy(pred, target, reduction='none')

        if mask is not None:
            loss = loss * mask
            target = target * mask

        positive_index = (target == 1.0).float()
        negative_index = (target == 0.0).float()

        positive_count = int(positive_index.sum().item())
        if positive_count == 0:
            return loss.mean()

        negative_count = int(negative_index.sum().item())
        max_neg = int(positive_count * self.negative_ratio)
        negative_count = min(negative_count, max_neg)

        positive_loss = (loss * positive_index).sum()

        negative_loss_all = loss * negative_index
        if negative_count > 0:
            negative_loss_flat = negative_loss_all.reshape(-1)  # reshape 替代 view
            topk_vals, _ = torch.topk(negative_loss_flat, negative_count)
            negative_loss = topk_vals.sum()
        else:
            negative_loss = loss.new_tensor(0.0)

        balance_loss = (positive_loss + negative_loss) / (positive_count + negative_count + self.eps)
        return balance_loss


def make_length_mask(lengths, max_len=None, device=None):
    """
    根据每条序列的有效长度，生成 (B, T) 的 0/1 mask。
    lengths: (B,)  每条序列的有效帧数
    max_len: 序列最大长度，默认取 lengths.max()
    """
    if device is None:
        device = lengths.device
    if max_len is None:
        max_len = int(lengths.max().item())
    B = lengths.size(0)
    range_t = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
    mask = (range_t < lengths.unsqueeze(1)).float()             # (B, T)
    return mask


class BAMLoss(nn.Module):
    """
    单个 BAM 的帧级 loss：
      total_loss = spoof_CE + lambda_boundary * boundary_BalanceBCE
    """
    def __init__(
        self,
        lambda_boundary: float = 0.5,
        ce_weight=None,
        negative_ratio: float = 5.0,
    ):
        super(BAMLoss, self).__init__()
        self.lambda_boundary = lambda_boundary

        self.ce_loss_fn = MaskCrossEnrtopyLoss(weight=ce_weight)
        self.boundary_loss_fn = BalanceBCELoss(negative_ratio=negative_ratio)

    def forward(
        self,
        output: torch.Tensor,
        boundary: torch.Tensor,
        label_cls: torch.Tensor,
        label_boundary: torch.Tensor,
        len_cls: torch.Tensor,
        len_boundary: torch.Tensor,
    ):
        """
        output:   (B, T, 2)
        boundary: (B, T)
        label_cls:      (B, T)
        label_boundary: (B, T)
        len_cls:        (B,)
        len_boundary:   (B,)
        """
        B, T, C = output.size()
        assert C == 2, "output 最后一维必须是 2（bona/spoof）"

        # transpose 结果通常 non-contiguous，后续 loss 内部用 reshape 已可处理
        output_ce = output.transpose(1, 2)  # (B, 2, T)

        spoof_mask = make_length_mask(len_cls, max_len=T, device=output.device)

        spoof_loss = self.ce_loss_fn(
            pred=output_ce,
            target=label_cls.long(),
            mask=spoof_mask,
        )

        boundary = boundary.reshape_as(label_boundary).float()
        boundary_mask = make_length_mask(len_boundary, max_len=boundary.size(1), device=boundary.device)

        boundary_loss = self.boundary_loss_fn(
            pred=boundary,
            target=label_boundary.float(),
            mask=boundary_mask,
        )

        total_loss = spoof_loss + self.lambda_boundary * boundary_loss
        return total_loss, spoof_loss, boundary_loss


# ======================= Multi-head loss（UNet + 3×BAM） =======================

class BAMMultiHeadLoss(nn.Module):
    """
    Multi-head 部分的 utter-level + 分离 loss：

      total = λ_sepa * L_sepa
            + λ_mix      * CE(logits_mix,     label_mix)
            + λ_sp_ref   * CE(logits_sp_ref,  label_speech)
            + λ_env_ref  * CE(logits_env_ref, label_env)
            + λ_sp_hat   * CE(logits_sp_hat,  label_speech)
            + λ_env_hat  * CE(logits_env_hat, label_env)

    其中：
      - speech_hat / env_hat: UNet 分离出来的波形 (B, T)
      - ref_speech / ref_env: 真值分量波形 (B, T)
      - logits_xxx:           (B, 2)，由 frame logits 平均得到
      - label_xxx:            (B,) 0/1 标签
      - joint = False：只训 mix/sp_ref/env_ref（以及分离 MSE）
      - joint = True：再打开 sp_hat/env_hat 的损失
    """
    def __init__(
        self,
        lambda_sepa: float = 10.0,
        lambda_mix: float = 1.0,
        lambda_sp_ref: float = 1.0,
        lambda_env_ref: float = 1.0,
        lambda_sp_hat: float = 1.0,
        lambda_env_hat: float = 1.0,
        class_weight_mix=None,
        class_weight_comp=None,
    ):
        super(BAMMultiHeadLoss, self).__init__()

        self.lambda_sepa = lambda_sepa
        self.lambda_mix = lambda_mix
        self.lambda_sp_ref = lambda_sp_ref
        self.lambda_env_ref = lambda_env_ref
        self.lambda_sp_hat = lambda_sp_hat
        self.lambda_env_hat = lambda_env_hat

        self.mse = nn.MSELoss()

        self.ce_mix = nn.CrossEntropyLoss(weight=class_weight_mix)
        self.ce_comp = nn.CrossEntropyLoss(weight=class_weight_comp)

    def _ce_branch(self, logits, labels, use_mix_head=False):
        """
        logits: (B, 2)，labels: (B,)
        use_mix_head=True 时，用 mix 的 weight；否则用 comp head 的 weight。
        """
        if logits is None:
            return logits.new_tensor(0.0)
        labels = labels.long()
        if use_mix_head:
            return self.ce_mix(logits, labels)
        else:
            return self.ce_comp(logits, labels)

    def forward(
        self,
        speech_hat: torch.Tensor,
        env_hat: torch.Tensor,
        ref_speech: torch.Tensor,
        ref_env: torch.Tensor,
        logits_mix: torch.Tensor,
        logits_sp_ref: torch.Tensor,
        logits_env_ref: torch.Tensor,
        logits_sp_hat: torch.Tensor,
        logits_env_hat: torch.Tensor,
        label_speech: torch.Tensor,
        label_env: torch.Tensor,
        label_mix: torch.Tensor,
        joint: bool = True,
    ):
        """
        返回：
          total_loss, {
              "loss_sepa", "loss_mix",
              "loss_sp_ref", "loss_env_ref",
              "loss_sp_hat", "loss_env_hat",
          }
        """
        loss_sepa = self.mse(speech_hat, ref_speech) + self.mse(env_hat, ref_env)

        loss_mix     = self._ce_branch(logits_mix,     label_mix,    use_mix_head=True)
        loss_sp_ref  = self._ce_branch(logits_sp_ref,  label_speech, use_mix_head=False)
        loss_env_ref = self._ce_branch(logits_env_ref, label_env,    use_mix_head=False)

        if joint:
            loss_sp_hat  = self._ce_branch(logits_sp_hat,  label_speech, use_mix_head=False)
            loss_env_hat = self._ce_branch(logits_env_hat, label_env,    use_mix_head=False)
        else:
            loss_sp_hat  = logits_sp_hat.new_tensor(0.0)
            loss_env_hat = logits_env_hat.new_tensor(0.0)

        total_loss = (
            self.lambda_sepa      * loss_sepa
            + self.lambda_mix     * loss_mix
            + self.lambda_sp_ref  * loss_sp_ref
            + self.lambda_env_ref * loss_env_ref
            + self.lambda_sp_hat  * loss_sp_hat
            + self.lambda_env_hat * loss_env_hat
        )

        loss_dict = {
            "loss_sepa":    loss_sepa,
            "loss_mix":     loss_mix,
            "loss_sp_ref":  loss_sp_ref,
            "loss_env_ref": loss_env_ref,
            "loss_sp_hat":  loss_sp_hat,
            "loss_env_hat": loss_env_hat,
        }

        return total_loss, loss_dict
