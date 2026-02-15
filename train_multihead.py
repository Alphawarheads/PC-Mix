# train_bam_multihead.py

import os
import argparse
import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from utils import (
    import_class,
    import_class_from_path,
    save_running_script,
    compute_eer,
    computer_precision_recall_fscore,
    Attribution_Config,
    cut_according_length,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from models.bam_multihead_loss import (
    BAMLoss,
    BAMMultiHeadLoss,
)


class LightingMultiHeadBAMWrapper(L.LightningModule):
    """
    UNet + 3×BAM 的 multi-head 训练包装：

    - 对 3 个 BAM 分支（mix / speech_hat / env_hat）都做帧级 spoof + boundary 监督（BAMLoss）
    - 同时叠加 UNet + multi-head 的 utter-level 任务（BAMMultiHeadLoss）
    - frame 级评估：mix / speech_hat / env_hat 三个域都做 spoof+boundary 的 EER/F1/acc
    - utter-level：三路（mix / speech_hat / env_hat）EER/F1/acc

    模型 forward 约定（BAMMultiHeadModel）：

        (
          frame_mix, boundary_mix,
          frame_sp_hat, boundary_sp_hat,
          frame_env_hat, boundary_env_hat,
          speech_hat, env_hat,
          logits_mix, logits_sp_ref, logits_env_ref,
          logits_sp_hat, logits_env_hat,
        ) = model(mix_input, ref_speech, ref_env)

    batch 结构约定（DataModule 需对齐）：

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

    def __init__(self, args, config):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.config = config

        # ========= 模型加载 =========
        model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.py')
        if os.path.exists(model_file_path):
            model_cls, package_path = import_class_from_path(args.model, model_file_path)
        else:
            model_cls, package_path = import_class(args.model)

        self.model = model_cls(args, config)
        self.model_file_path = package_path
        print(f'Load model file from {package_path}.')

        # ========= 3 个 BAM 分支共用的帧级 loss（内部有 CE + BalanceBCE） =========
        self.bam_loss_fn = BAMLoss(
            lambda_boundary=0.5,
            ce_weight=None,
            negative_ratio=5.0,
        )

        # ========= Multi-head loss（包含分离 MSE + 各 utter-level head 分类） =========
        self.multihead_loss = BAMMultiHeadLoss(
            lambda_sepa=args.lambda_sepa,
            lambda_mix=args.lambda_mix,
            lambda_sp_ref=args.lambda_sp_ref,
            lambda_env_ref=args.lambda_env_ref,
            lambda_sp_hat=args.lambda_sp_hat,
            lambda_env_hat=args.lambda_env_hat,
            class_weight_mix=None,
            class_weight_comp=None,
        )

        # ========= frame-level EER 相关缓存（mix） =========
        self.train_pred_labels = []
        self.train_step_outputs = []
        self.test_pred_labels = []
        self.test_step_outputs = []
        self.validate_pred_labels = []
        self.validate_step_outputs = []
        self.utt_id_list = []

        self.b_train_pred_labels = []
        self.b_train_step_outputs = []
        self.b_validate_pred_labels = []
        self.b_validate_step_outputs = []
        self.b_test_pred_labels = []
        self.b_test_step_outputs = []

        # ========= 新增：speech_hat / env_hat 的 frame-level 缓存 =========
        # spoof logits
        self.sp_train_step_outputs = []
        self.sp_train_pred_labels = []
        self.sp_validate_step_outputs = []
        self.sp_validate_pred_labels = []
        self.sp_test_step_outputs = []
        self.sp_test_pred_labels = []

        self.env_train_step_outputs = []
        self.env_train_pred_labels = []
        self.env_validate_step_outputs = []
        self.env_validate_pred_labels = []
        self.env_test_step_outputs = []
        self.env_test_pred_labels = []

        # boundary prob
        self.b_sp_train_step_outputs = []
        self.b_sp_train_pred_labels = []
        self.b_sp_validate_step_outputs = []
        self.b_sp_validate_pred_labels = []
        self.b_sp_test_step_outputs = []
        self.b_sp_test_pred_labels = []

        self.b_env_train_step_outputs = []
        self.b_env_train_pred_labels = []
        self.b_env_validate_step_outputs = []
        self.b_env_validate_pred_labels = []
        self.b_env_test_step_outputs = []
        self.b_env_test_pred_labels = []

        # ========= utter-level multi-head 评估缓存（mix / speech / env） =========
        self.mix_utt_logits = []
        self.mix_utt_labels = []
        self.sp_utt_logits = []
        self.sp_utt_labels = []
        self.env_utt_logits = []
        self.env_utt_labels = []

    def setup(self, stage: str) -> None:
        # 保存运行脚本
        save_running_script(
            os.path.abspath(__file__),
            f'{trainer.logger.root_dir}/version_{trainer.logger.version}/run.py'
        )
        save_running_script(
            os.path.abspath(self.model_file_path),
            f'{trainer.logger.root_dir}/version_{trainer.logger.version}/model.py'
        )

        # 日志文件
        if self.local_rank == 0:
            self.console_logger = logging.getLogger(f'lightning.pytorch.{stage}')
            file_handler = logging.FileHandler(
                f'{trainer.logger.root_dir}/version_{trainer.logger.version}/{stage}.log'
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'))
            self.console_logger.addHandler(file_handler)
            self.console_logger.info(f'Start {stage}.')

    def training_step(self, batch, batch_idx):
        (utt_id,
         mix_input, ref_speech, ref_env,
         ori_label, boundary_label,
         ori_label_length, boundary_length,
         sp_label, sp_boundary_label,
         sp_label_length, sp_boundary_length,
         env_label, env_boundary_label,
         env_label_length, env_boundary_length,
         label_speech_utt, label_env_utt, label_mix_utt) = batch

        # ========= 1. 前向 =========
        (frame_mix, boundary_mix,
         frame_sp_hat, boundary_sp_hat,
         frame_env_hat, boundary_env_hat,
         speech_hat, env_hat,
         logits_mix, logits_sp_ref, logits_env_ref,
         logits_sp_hat, logits_env_hat) = self.model(mix_input, ref_speech, ref_env)

        # ========= 2. 三个 BAM 分支的帧级 loss =========
        mix_total, mix_spoof_loss, mix_boundary_loss = self.bam_loss_fn(
            output=frame_mix,
            boundary=boundary_mix,
            label_cls=ori_label,
            label_boundary=boundary_label,
            len_cls=ori_label_length,
            len_boundary=boundary_length,
        )

        sp_total, sp_spoof_loss, sp_boundary_loss = self.bam_loss_fn(
            output=frame_sp_hat,
            boundary=boundary_sp_hat,
            label_cls=sp_label,
            label_boundary=sp_boundary_label,
            len_cls=sp_label_length,
            len_boundary=sp_boundary_length,
        )

        env_total, env_spoof_loss, env_boundary_loss = self.bam_loss_fn(
            output=frame_env_hat,
            boundary=boundary_env_hat,
            label_cls=env_label,
            label_boundary=env_boundary_label,
            len_cls=env_label_length,
            len_boundary=env_boundary_length,
        )

        bam_frame_total = mix_total + sp_total + env_total

        # ========= 3. Multi-head utter-level loss（包含分离 MSE） =========
        joint = (self.current_epoch + 1) >= self.args.joint_start_epoch

        mh_total, mh_dict = self.multihead_loss(
            speech_hat=speech_hat,
            env_hat=env_hat,
            ref_speech=ref_speech,
            ref_env=ref_env,
            logits_mix=logits_mix,
            logits_sp_ref=logits_sp_ref,
            logits_env_ref=logits_env_ref,
            logits_sp_hat=logits_sp_hat,
            logits_env_hat=logits_env_hat,
            label_speech=label_speech_utt,
            label_env=label_env_utt,
            label_mix=label_mix_utt,
            joint=joint,
        )

        total_loss = bam_frame_total + self.args.lambda_mh * mh_total

        # ========= 4. logging =========
        self.log('train_loss', total_loss.item(), prog_bar=True, on_epoch=True, sync_dist=True)

        # BAM 帧级 loss
        self.log('mix_bam_total', mix_total.item(), sync_dist=True)
        self.log('mix_spoof_loss', mix_spoof_loss.item(), sync_dist=True)
        self.log('mix_boundary_loss', mix_boundary_loss.item(), sync_dist=True)

        self.log('sp_bam_total', sp_total.item(), sync_dist=True)
        self.log('sp_spoof_loss', sp_spoof_loss.item(), sync_dist=True)
        self.log('sp_boundary_loss', sp_boundary_loss.item(), sync_dist=True)

        self.log('env_bam_total', env_total.item(), sync_dist=True)
        self.log('env_spoof_loss', env_spoof_loss.item(), sync_dist=True)
        self.log('env_boundary_loss', env_boundary_loss.item(), sync_dist=True)

        # Multi-head loss
        self.log('mh_total_loss', mh_total.item(), sync_dist=True)
        self.log('mh_sepa_loss', mh_dict['loss_sepa'].item(), sync_dist=True)
        self.log('mh_mix_loss', mh_dict['loss_mix'].item(), sync_dist=True)
        self.log('mh_sp_ref_loss', mh_dict['loss_sp_ref'].item(), sync_dist=True)
        self.log('mh_env_ref_loss', mh_dict['loss_env_ref'].item(), sync_dist=True)
        self.log('mh_sp_hat_loss', mh_dict['loss_sp_hat'].item(), sync_dist=True)
        self.log('mh_env_hat_loss', mh_dict['loss_env_hat'].item(), sync_dist=True)
        self.log('joint_phase', float(joint), prog_bar=True, sync_dist=True)

        # ========= 5. frame-level 统计（train：mix/speech/env 三域都缓存） =========
        # mix
        pred_mix, lab_mix = cut_according_length(frame_mix.detach(), ori_label, ori_label_length)
        b_pred_mix, b_lab_mix = cut_according_length(boundary_mix.detach(), boundary_label, boundary_length)
        self.train_step_outputs.extend(pred_mix)
        self.train_pred_labels.extend(lab_mix)
        self.b_train_step_outputs.extend(b_pred_mix)
        self.b_train_pred_labels.extend(b_lab_mix)

        # speech_hat
        pred_sp, lab_sp = cut_according_length(frame_sp_hat.detach(), sp_label, sp_label_length)
        b_pred_sp, b_lab_sp = cut_according_length(boundary_sp_hat.detach(), sp_boundary_label, sp_boundary_length)
        self.sp_train_step_outputs.extend(pred_sp)
        self.sp_train_pred_labels.extend(lab_sp)
        self.b_sp_train_step_outputs.extend(b_pred_sp)
        self.b_sp_train_pred_labels.extend(b_lab_sp)

        # env_hat
        pred_env, lab_env = cut_according_length(frame_env_hat.detach(), env_label, env_label_length)
        b_pred_env, b_lab_env = cut_according_length(boundary_env_hat.detach(), env_boundary_label, env_boundary_length)
        self.env_train_step_outputs.extend(pred_env)
        self.env_train_pred_labels.extend(lab_env)
        self.b_env_train_step_outputs.extend(b_pred_env)
        self.b_env_train_pred_labels.extend(b_lab_env)

        self.utt_id_list.extend(utt_id)

        # ========= 6. utter-level multi-head 统计（train） =========
        self.mix_utt_logits.extend(logits_mix.detach().cpu().tolist())
        self.mix_utt_labels.extend(label_mix_utt.detach().cpu().tolist())

        self.sp_utt_logits.extend(logits_sp_hat.detach().cpu().tolist())
        self.sp_utt_labels.extend(label_speech_utt.detach().cpu().tolist())

        self.env_utt_logits.extend(logits_env_hat.detach().cpu().tolist())
        self.env_utt_labels.extend(label_env_utt.detach().cpu().tolist())

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.base_lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def evaluation_run_model(self, batch, type: str):
        """
        评估阶段：
          - frame-level：mix / speech_hat / env_hat 三个域都做 spoof+boundary 的 EER/F1/acc
          - validate_loss：只对 mix 分支算 masked BAMLoss（不污染 checkpoint metric）
          - utter-level：收集 mix / speech_hat / env_hat 的 logits + label
        """
        # mix collectors (保持原 key)
        outputs_collector = getattr(self, f'{type}_step_outputs')
        labels_collector = getattr(self, f'{type}_pred_labels')
        b_outputs_collector = getattr(self, f'b_{type}_step_outputs')
        b_labels_collector = getattr(self, f'b_{type}_pred_labels')

        # speech collectors
        sp_outputs_collector = getattr(self, f'sp_{type}_step_outputs')
        sp_labels_collector = getattr(self, f'sp_{type}_pred_labels')
        b_sp_outputs_collector = getattr(self, f'b_sp_{type}_step_outputs')
        b_sp_labels_collector = getattr(self, f'b_sp_{type}_pred_labels')

        # env collectors
        env_outputs_collector = getattr(self, f'env_{type}_step_outputs')
        env_labels_collector = getattr(self, f'env_{type}_pred_labels')
        b_env_outputs_collector = getattr(self, f'b_env_{type}_step_outputs')
        b_env_labels_collector = getattr(self, f'b_env_{type}_pred_labels')

        (utt_id,
         mix_input, ref_speech, ref_env,
         ori_label, boundary_label,
         ori_label_length, boundary_length,
         sp_label, sp_boundary_label,
         sp_label_length, sp_boundary_length,
         env_label, env_boundary_label,
         env_label_length, env_boundary_length,
         label_speech_utt, label_env_utt, label_mix_utt) = batch

        # pad mix_input 到 mix label 对应长度（只在需要时 pad）
        scale = int(self.args.resolution * self.args.samplerate)
        target_len = ori_label.size(1) * scale
        cur_len = mix_input.size(1)
        if cur_len < target_len:
            mix_input = F.pad(mix_input, (0, target_len - cur_len))

        (frame_mix, boundary_mix,
         frame_sp_hat, boundary_sp_hat,
         frame_env_hat, boundary_env_hat,
         speech_hat, env_hat,
         logits_mix, logits_sp_ref, logits_env_ref,
         logits_sp_hat, logits_env_hat) = self.model(mix_input, ref_speech, ref_env)

        # validate loss（只用 mix 分支，且带 mask）
        if type == 'validate':
            mix_total, mix_spoof_loss, mix_boundary_loss = self.bam_loss_fn(
                output=frame_mix,
                boundary=boundary_mix,
                label_cls=ori_label,
                label_boundary=boundary_label,
                len_cls=ori_label_length,
                len_boundary=boundary_length,
            )
            self.log('validate_loss', mix_total.item(), sync_dist=True)
            self.log('validate_spoof_loss', mix_spoof_loss.item(), sync_dist=True)
            self.log('validate_boundary_loss', mix_boundary_loss.item(), sync_dist=True)

        # ===== frame-level 缓存：三域都用 cut_according_length 去 padding =====
        # mix
        pred_mix, lab_mix = cut_according_length(frame_mix.detach(), ori_label, ori_label_length)
        b_pred_mix, b_lab_mix = cut_according_length(boundary_mix.detach(), boundary_label, boundary_length)
        outputs_collector.extend(pred_mix)
        labels_collector.extend(lab_mix)
        b_outputs_collector.extend(b_pred_mix)
        b_labels_collector.extend(b_lab_mix)

        # speech_hat
        pred_sp, lab_sp = cut_according_length(frame_sp_hat.detach(), sp_label, sp_label_length)
        b_pred_sp, b_lab_sp = cut_according_length(boundary_sp_hat.detach(), sp_boundary_label, sp_boundary_length)
        sp_outputs_collector.extend(pred_sp)
        sp_labels_collector.extend(lab_sp)
        b_sp_outputs_collector.extend(b_pred_sp)
        b_sp_labels_collector.extend(b_lab_sp)

        # env_hat
        pred_env, lab_env = cut_according_length(frame_env_hat.detach(), env_label, env_label_length)
        b_pred_env, b_lab_env = cut_according_length(boundary_env_hat.detach(), env_boundary_label, env_boundary_length)
        env_outputs_collector.extend(pred_env)
        env_labels_collector.extend(lab_env)
        b_env_outputs_collector.extend(b_pred_env)
        b_env_labels_collector.extend(b_lab_env)

        self.utt_id_list.extend(utt_id)

        # ===== utter-level 缓存 =====
        self.mix_utt_logits.extend(logits_mix.detach().cpu().tolist())
        self.mix_utt_labels.extend(label_mix_utt.detach().cpu().tolist())

        self.sp_utt_logits.extend(logits_sp_hat.detach().cpu().tolist())
        self.sp_utt_labels.extend(label_speech_utt.detach().cpu().tolist())

        self.env_utt_logits.extend(logits_env_hat.detach().cpu().tolist())
        self.env_utt_labels.extend(label_env_utt.detach().cpu().tolist())

    def evaluation_on_epoch_end(self, type):
        # mix frame cache
        outputs = getattr(self, f'{type}_step_outputs')
        labels = getattr(self, f'{type}_pred_labels')
        b_outputs = getattr(self, f'b_{type}_step_outputs')
        b_labels = getattr(self, f'b_{type}_pred_labels')

        # speech frame cache
        sp_outputs = getattr(self, f'sp_{type}_step_outputs')
        sp_labels = getattr(self, f'sp_{type}_pred_labels')
        b_sp_outputs = getattr(self, f'b_sp_{type}_step_outputs')
        b_sp_labels = getattr(self, f'b_sp_{type}_pred_labels')

        # env frame cache
        env_outputs = getattr(self, f'env_{type}_step_outputs')
        env_labels = getattr(self, f'env_{type}_pred_labels')
        b_env_outputs = getattr(self, f'b_env_{type}_step_outputs')
        b_env_labels = getattr(self, f'b_env_{type}_pred_labels')

        utt_ids = self.utt_id_list

        # ===== frame-level spoof (mix) =====
        if len(outputs) > 0:
            frame_preds = torch.tensor([i for utt in outputs for i in utt]).detach().cpu()
            frame_labels = torch.tensor([i for utt in labels for i in utt]).detach().cpu()
            eer, threshold = compute_eer(frame_preds[:, 1], frame_labels)
            accuracy, precision, recall, fbeta_score = computer_precision_recall_fscore(
                frame_preds.argmax(dim=-1), frame_labels
            )

            # 保持原 key：{type}/{type}_xxx
            self.log(f'{type}/{type}_F1', fbeta_score, sync_dist=True)
            self.log(f'{type}/{type}_eer', eer, sync_dist=True)
            self.log(f'{type}/{type}_acc', accuracy, sync_dist=True)

            if self.local_rank == 0:
                self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} mix eer {eer}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} mix acc {accuracy}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} mix precision {precision}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} mix recall {recall}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} mix F1 {fbeta_score}')
                self.console_logger.info('---------------------------------------------------------')

            if self.args.test_only and type == 'test':
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result.txt'), 'a') as result_file:
                    result_file.write(f'Checkpoint :{self.args.checkpoint} \n')
                    result_file.write(f'Frame-level mix EER :{eer * 100}% \n')
                    result_file.write(f'Frame-level mix F1 :{fbeta_score} \n')
                    result_file.write(f'Frame-level mix Precision :{precision} \n')
                    result_file.write(f'Frame-level mix Recall :{recall} \n')
                    result_file.write(
                        f'Test log :{trainer.logger.root_dir}/version_{trainer.logger.version} \n'
                    )
                    result_file.write(f'\n')

        # ===== frame-level boundary (mix) =====
        if len(b_outputs) > 0:
            b_frame_preds = torch.tensor([i for utt in b_outputs for i in utt]).detach().cpu()
            b_frame_labels = torch.tensor([i for utt in b_labels for i in utt]).detach().cpu()
            eer_b, threshold_b = compute_eer(b_frame_preds, b_frame_labels)
            accuracy_b, precision_b, recall_b, fbeta_score_b = computer_precision_recall_fscore(
                torch.where(b_frame_preds > 0.5, 1, 0), b_frame_labels
            )

            # 保持原 key：{type}/b_{type}_xxx
            self.log(f'{type}/b_{type}_F1', fbeta_score_b, sync_dist=True)
            self.log(f'{type}/b_{type}_eer', eer_b, sync_dist=True)
            self.log(f'{type}/b_{type}_acc', accuracy_b, sync_dist=True)

            if self.local_rank == 0:
                self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} mix eer {eer_b}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} mix acc {accuracy_b}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} mix precision {precision_b}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} mix recall {recall_b}')
                self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} mix F1 {fbeta_score_b}')
                self.console_logger.info('---------------------------------------------------------')

        # ===== frame-level spoof (speech_hat) =====
        if len(sp_outputs) > 0:
            sp_frame_preds = torch.tensor([i for utt in sp_outputs for i in utt]).detach().cpu()
            sp_frame_labels = torch.tensor([i for utt in sp_labels for i in utt]).detach().cpu()
            eer_sp_f, _ = compute_eer(sp_frame_preds[:, 1], sp_frame_labels)
            acc_sp_f, prec_sp_f, rec_sp_f, f1_sp_f = computer_precision_recall_fscore(
                sp_frame_preds.argmax(dim=-1), sp_frame_labels
            )
            self.log(f'{type}/sp_{type}_eer', eer_sp_f, sync_dist=True)
            self.log(f'{type}/sp_{type}_F1', f1_sp_f, sync_dist=True)
            self.log(f'{type}/sp_{type}_acc', acc_sp_f, sync_dist=True)

        # ===== frame-level boundary (speech_hat) =====
        if len(b_sp_outputs) > 0:
            b_sp_preds = torch.tensor([i for utt in b_sp_outputs for i in utt]).detach().cpu()
            b_sp_labs = torch.tensor([i for utt in b_sp_labels for i in utt]).detach().cpu()
            eer_b_sp, _ = compute_eer(b_sp_preds, b_sp_labs)
            acc_b_sp, prec_b_sp, rec_b_sp, f1_b_sp = computer_precision_recall_fscore(
                torch.where(b_sp_preds > 0.5, 1, 0), b_sp_labs
            )
            self.log(f'{type}/b_sp_{type}_eer', eer_b_sp, sync_dist=True)
            self.log(f'{type}/b_sp_{type}_F1', f1_b_sp, sync_dist=True)
            self.log(f'{type}/b_sp_{type}_acc', acc_b_sp, sync_dist=True)

        # ===== frame-level spoof (env_hat) =====
        if len(env_outputs) > 0:
            env_frame_preds = torch.tensor([i for utt in env_outputs for i in utt]).detach().cpu()
            env_frame_labels = torch.tensor([i for utt in env_labels for i in utt]).detach().cpu()
            eer_env_f, _ = compute_eer(env_frame_preds[:, 1], env_frame_labels)
            acc_env_f, prec_env_f, rec_env_f, f1_env_f = computer_precision_recall_fscore(
                env_frame_preds.argmax(dim=-1), env_frame_labels
            )
            self.log(f'{type}/env_{type}_eer', eer_env_f, sync_dist=True)
            self.log(f'{type}/env_{type}_F1', f1_env_f, sync_dist=True)
            self.log(f'{type}/env_{type}_acc', acc_env_f, sync_dist=True)

        # ===== frame-level boundary (env_hat) =====
        if len(b_env_outputs) > 0:
            b_env_preds = torch.tensor([i for utt in b_env_outputs for i in utt]).detach().cpu()
            b_env_labs = torch.tensor([i for utt in b_env_labels for i in utt]).detach().cpu()
            eer_b_env, _ = compute_eer(b_env_preds, b_env_labs)
            acc_b_env, prec_b_env, rec_b_env, f1_b_env = computer_precision_recall_fscore(
                torch.where(b_env_preds > 0.5, 1, 0), b_env_labs
            )
            self.log(f'{type}/b_env_{type}_eer', eer_b_env, sync_dist=True)
            self.log(f'{type}/b_env_{type}_F1', f1_b_env, sync_dist=True)
            self.log(f'{type}/b_env_{type}_acc', acc_b_env, sync_dist=True)

        # ===== utter-level multi-head（mix / speech / env） =====
        if len(self.mix_utt_logits) > 0:
            mix_logits = torch.tensor(self.mix_utt_logits)   # (N, 2)
            mix_labels = torch.tensor(self.mix_utt_labels)   # (N,)
            mix_scores = mix_logits[:, 1]
            mix_pred = mix_logits.argmax(dim=-1)

            eer_mix, thr_mix = compute_eer(mix_scores, mix_labels)
            acc_mix, prec_mix, rec_mix, f1_mix = computer_precision_recall_fscore(mix_pred, mix_labels)

            self.log(f'{type}/utt_mix_eer', eer_mix, sync_dist=True)
            self.log(f'{type}/utt_mix_F1', f1_mix, sync_dist=True)
            self.log(f'{type}/utt_mix_acc', acc_mix, sync_dist=True)

        if len(self.sp_utt_logits) > 0:
            sp_logits = torch.tensor(self.sp_utt_logits)
            sp_labels_u = torch.tensor(self.sp_utt_labels)
            sp_scores = sp_logits[:, 1]
            sp_pred = sp_logits.argmax(dim=-1)

            eer_sp_u, thr_sp_u = compute_eer(sp_scores, sp_labels_u)
            acc_sp_u, prec_sp_u, rec_sp_u, f1_sp_u = computer_precision_recall_fscore(sp_pred, sp_labels_u)

            self.log(f'{type}/utt_speech_eer', eer_sp_u, sync_dist=True)
            self.log(f'{type}/utt_speech_F1', f1_sp_u, sync_dist=True)
            self.log(f'{type}/utt_speech_acc', acc_sp_u, sync_dist=True)

        if len(self.env_utt_logits) > 0:
            env_logits = torch.tensor(self.env_utt_logits)
            env_labels_u = torch.tensor(self.env_utt_labels)
            env_scores = env_logits[:, 1]
            env_pred = env_logits.argmax(dim=-1)

            eer_env_u, thr_env_u = compute_eer(env_scores, env_labels_u)
            acc_env_u, prec_env_u, rec_env_u, f1_env_u = computer_precision_recall_fscore(env_pred, env_labels_u)

            self.log(f'{type}/utt_env_eer', eer_env_u, sync_dist=True)
            self.log(f'{type}/utt_env_F1', f1_env_u, sync_dist=True)
            self.log(f'{type}/utt_env_acc', acc_env_u, sync_dist=True)

            if self.args.test_only and type == 'test':
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result.txt'), 'a') as result_file:
                    # 这里 mix/sp/env 的 utter-level EER 只有在对应 logits 存在时才写入
                    # 防止变量未定义
                    if len(self.mix_utt_logits) > 0:
                        result_file.write(f'Utter-level mix EER :{eer_mix * 100}% \n')
                    if len(self.sp_utt_logits) > 0:
                        result_file.write(f'Utter-level speech EER :{eer_sp_u * 100}% \n')
                    if len(self.env_utt_logits) > 0:
                        result_file.write(f'Utter-level env EER :{eer_env_u * 100}% \n')
                    result_file.write('\n')

        # ===== 清空缓存 =====
        # mix
        outputs.clear()
        labels.clear()
        b_outputs.clear()
        b_labels.clear()

        # speech/env frame caches
        sp_outputs.clear()
        sp_labels.clear()
        b_sp_outputs.clear()
        b_sp_labels.clear()

        env_outputs.clear()
        env_labels.clear()
        b_env_outputs.clear()
        b_env_labels.clear()

        utt_ids.clear()

        # utter-level
        self.mix_utt_logits.clear()
        self.mix_utt_labels.clear()
        self.sp_utt_logits.clear()
        self.sp_utt_labels.clear()
        self.env_utt_logits.clear()
        self.env_utt_labels.clear()

    def validation_step(self, batch, batch_idx):
        self.evaluation_run_model(batch, type='validate')

    def test_step(self, batch, batch_idx):
        self.evaluation_run_model(batch, type='test')

    def on_validation_epoch_end(self):
        self.evaluation_on_epoch_end(type='validate')

    def on_test_epoch_end(self):
        self.evaluation_on_epoch_end(type='test')

    def on_train_epoch_end(self):
        self.evaluation_on_epoch_end(type='train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, help="dataset module",
                        default='dataset.partialspoofMul.PartialSpoofDataModule')
    parser.add_argument('--model', type=str, help="model module",
                        default='models.BAM_multihead.BAMMultiHeadModel')

    parser.add_argument('--train_root', type=str, help="train data path (mix)", default='data/raw/train')
    parser.add_argument('--dev_root', type=str, help="validate data path (mix)",
                        default='/DATA1/zhangzs/partComSpoof/BAM-master/data_partial_eval/raw/partial_env')
    parser.add_argument('--eval_root', type=str, help="test data path (mix)", default='data/raw/eval')

    parser.add_argument('--ref_speech_root', type=str, help="reference clean speech path", default=None)
    parser.add_argument('--ref_env_root', type=str, help="reference background/env path", default=None)

    parser.add_argument('--label_root', type=str, default='./data', help="segment label path")

    # training configuration
    parser.add_argument('--max_epochs', type=int, default=50, help='max train epoch.')
    parser.add_argument('--batch_size', type=int, default=8, help='train dataloader batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='train dataloader of num workers')
    parser.add_argument('--base_lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--samplerate', type=int, default=16000, help="samplerate")
    parser.add_argument('--resolution', type=float, default=0.16, help="segment label resolution.")
    parser.add_argument('--input_maxlength', type=int, default=None, help="unused")
    parser.add_argument('--input_minlength', type=int, default=None, help="min length of label or audio")
    parser.add_argument('--label_maxlength', type=int, default=25, help="max length of label or audio")
    parser.add_argument('--pad_mode', type=str, default='label', help='how to pad data')

    parser.add_argument('--gpu', type=list, default=[0], help="gpu index")
    parser.add_argument('--test_only', action='store_true', help="test model")
    parser.add_argument('--exp_name', type=str, default='bam_multihead', help="experiment name.")
    parser.add_argument('--validate_interval', type=int, default=3, help="do validate every n epochs")

    # model configuration
    parser.add_argument('--checkpoint', type=str, help='model checkpoint',
                        default='/public/home/qinxy/zhangzs/Partial_spoof/BAM-master/bam_checkpoint/model.ckpt')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training')

    # multi-head & two-stage 参数
    parser.add_argument('--joint_start_epoch', type=int, default=5,
                        help='epoch to start joint training (UNet + BAM heads on separated signals)')
    parser.add_argument('--lambda_mh', type=float, default=1.0,
                        help='overall weight for multi-head loss')
    parser.add_argument('--lambda_sepa', type=float, default=10.0,
                        help='weight for separation MSE in multi-head loss')
    parser.add_argument('--lambda_mix', type=float, default=1.0,
                        help='weight for mix classification loss')
    parser.add_argument('--lambda_sp_ref', type=float, default=1.0,
                        help='weight for speech ref classification loss')
    parser.add_argument('--lambda_env_ref', type=float, default=1.0,
                        help='weight for env ref classification loss')
    parser.add_argument('--lambda_sp_hat', type=float, default=1.0,
                        help='weight for speech hat classification loss')
    parser.add_argument('--lambda_env_hat', type=float, default=1.0,
                        help='weight for env hat classification loss')

    args = parser.parse_args()
    L.seed_everything(42, workers=True)

    # model config
    with open(f'config/{args.exp_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = Attribution_Config(**config)
    print(config.ssl_ckpt)

    if hasattr(args, 'checkpoint') and (args.test_only or args.continue_train):
        print(f'Load model from {args.checkpoint}.')
        args.checkpoint = os.readlink(args.checkpoint) if os.path.islink(args.checkpoint) else args.checkpoint
        model = LightingMultiHeadBAMWrapper.load_from_checkpoint(
            args.checkpoint, map_location='cpu', args=args, config=config
        )
    else:
        model = LightingMultiHeadBAMWrapper(args, config)
        print('Train model from scratch.')

    # define dataset
    dataset_cls, _ = import_class(args.dataset)
    Lightning_dataset = dataset_cls(args)

    checkponint_callback = ModelCheckpoint(
        filename='{epoch}-{validate_loss:.5f}',
        every_n_epochs=1,
        save_top_k=-1,
        save_weights_only=True,
        enable_version_counter=True,
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        accelerator='gpu',
        devices=args.gpu,
        max_epochs=args.max_epochs,
        strategy='auto' if len(args.gpu) == 1 else 'ddp_find_unused_parameters_true',
        logger=TensorBoardLogger(
            save_dir=f'exp/{args.exp_name}/test' if args.test_only else f'exp/{args.exp_name}/train'
        ),
        check_val_every_n_epoch=args.validate_interval,
        callbacks=checkponint_callback,
    )

    if not args.test_only:
        trainer.fit(model=model, datamodule=Lightning_dataset)
        print('Train finish.')
    else:
        trainer.test(model=model, datamodule=Lightning_dataset)
        print('Test finish.')
