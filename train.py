import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import random

# -----------------------------
# utils
# -----------------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()

import pytorch_model_summary
from torchinfo import summary
import torchsummary


# -----------------------------
# 内置简洁版 FDL（当 config 未提供现成 fdl_loss 接口时兜底）
# -----------------------------
def _fdl_sw_over_sb(D: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    D: (B, N, C) —— 类嵌入（该层最后一次交互）
    目标：最小化 Sw / (Sb + eps)
    """
    B, N, C = D.shape
    class_means = D.mean(dim=0)  # (N,C)
    # 类内散度
    diff_intra = D - class_means.unsqueeze(0)          # (B,N,C)
    Sw = (diff_intra ** 2).sum(dim=2).mean()           # scalar
    # 类间散度
    diff_inter = class_means.unsqueeze(1) - class_means.unsqueeze(0)  # (N,N,C)
    Sb_mat = (diff_inter ** 2).sum(dim=2)                              # (N,N)
    triu = torch.triu_indices(N, N, offset=1)
    Sb = Sb_mat[triu[0], triu[1]].mean()
    Sb = torch.clamp(Sb, min=1e-4)
    return Sw / (Sb + eps)


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss  # 你的 UnetFormerLoss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

        # ===== 读取开关与权重（带默认值）=====
        # 对比损失（模型内算出，外部只加权）
        self.has_contrastive_loss = getattr(config, "has_contrastive_loss", False)
        self.lambda_contrastive = getattr(config, "lambda_contrastive", 1.0)

        # 分层热力图深监督（H_list 用 UnetFormerLoss）
        self.use_heatmap_deep = getattr(config, "use_heatmap_deep", False)
        self.heatmap_beta = getattr(config, "heatmap_beta", [])
        self.lambda_heatmap = getattr(config, "lambda_heatmap", 1.0)

        # 类嵌入损失（优先使用已有接口）
        # 若 config 提供了 fdl_loss（callable/nn.Module），优先用它；
        # 否则使用内置 _fdl_sw_over_sb。
        self.fdl_loss_fn = getattr(config, "fdl_loss", None)
        self.use_fdl_final = getattr(config, "use_fdl_final", False)
        self.use_fdl_multi = getattr(config, "use_fdl_multi", False)
        self.lambda_fdl_final = getattr(config, "lambda_fdl_final", 1.0)
        self.lambda_fdl_multi = getattr(config, "lambda_fdl_multi", 1.0)
        self.fdl_alpha = getattr(config, "fdl_alpha", [])  # 分层权重
        self.fdl_eps = getattr(config, "fdl_eps", 1e-6)

    def forward(self, x):
        # 预测/验证仅用 net 前向
        seg_pre = self.net(x)
        return seg_pre

    # ---------------- training ----------------
    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']  # mask: (B,H,W) long

        # 期望训练态返回： (seg_logits, aux_logits, loss_con, H_list, D_list)
        prediction = self.net(img)

        # --- 解包，兼容不同返回形态 ---
        seg_logits, aux_logits, loss_con, H_list, D_list = None, None, None, [], []
        if isinstance(prediction, tuple):
            if len(prediction) >= 5:
                seg_logits, aux_logits, loss_con, H_list, D_list = prediction[:5]
            elif len(prediction) == 3:
                seg_logits, aux_logits, loss_con = prediction
            elif len(prediction) == 2:
                seg_logits, aux_logits = prediction
            else:
                seg_logits = prediction[0]
        else:
            seg_logits = prediction

        # --- 主/辅分支损失：沿用你的 UnetFormerLoss 接口 ---
        if aux_logits is not None:
            total_loss = self.loss((seg_logits, aux_logits), mask)
        else:
            total_loss = self.loss(seg_logits, mask)

        # --- 对比损失（可选，来自模型返回） ---
        if self.has_contrastive_loss and (loss_con is not None):
            total_loss = total_loss + self.lambda_contrastive * loss_con

        # --- 分层热力图深监督：H_list 逐层上采样 + 直接用 UnetFormerLoss（单 logits→只走 main 分支） ---
        if self.use_heatmap_deep and len(H_list) > 0:
            beta = self.heatmap_beta if len(self.heatmap_beta) == len(H_list) else [1.0] * len(H_list)
            loss_heat = 0.0
            for l, h_logits in enumerate(H_list):
                h_up = F.interpolate(h_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                loss_heat = loss_heat + beta[l] * self.loss(h_up, mask)
            total_loss = total_loss + self.lambda_heatmap * loss_heat

        # --- 类嵌入损失（FDL）：优先用已有接口；若无则用内置简洁实现 ---
        if (self.use_fdl_final or self.use_fdl_multi) and len(D_list) > 0:
            # 选择具体的 fdl 函数
            def _call_fdl_fn(D):
                if self.fdl_loss_fn is not None:
                    # 允许传 nn.Module 或函数；接口：loss = f(D, eps?) 或 f(D)
                    try:
                        return self.fdl_loss_fn(D)
                    except TypeError:
                        return self.fdl_loss_fn(D, eps=self.fdl_eps)
                else:
                    return _fdl_sw_over_sb(D, eps=self.fdl_eps)

            # 末层
            if self.use_fdl_final:
                loss_fdl_final = _call_fdl_fn(D_list[-1])
                total_loss = total_loss + self.lambda_fdl_final * loss_fdl_final

            # 分层
            if self.use_fdl_multi:
                alpha = self.fdl_alpha if len(self.fdl_alpha) == len(D_list) else [1.0] * len(D_list)
                loss_fdl_m = 0.0
                for l, Dl in enumerate(D_list):
                    loss_fdl_m = loss_fdl_m + alpha[l] * _call_fdl_fn(Dl)
                total_loss = total_loss + self.lambda_fdl_multi * loss_fdl_m

        # --- 指标：用主 seg_logits ---
        pred_mask = seg_logits.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

        return {"loss": total_loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'train_mIoU': mIoU,
                      'train_F1': F1,
                      'train_OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    # ---------------- validation ----------------
    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)  # eval 模式：模型只返回 seg_logits
        pre_mask = nn.Softmax(dim=1)(prediction).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'val_mIoU': mIoU,
                      'val_F1': F1,
                      'val_OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    # -------------- optim & loaders --------------
    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


# ---------------- training ----------------
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    # 仍然保留「根据 monitor 指标保存最优模型」的回调
    best_ckpt = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name  # 原命名规则
    )

    # 新增：每隔 5 个 epoch 额外保存一次（路径相同，命名加上 epoch）
    periodic_ckpt = ModelCheckpoint(
        dirpath=config.weights_path,
        filename=f"{config.weights_name}-e{{epoch:04d}}",  # 例：xxx-e0005.ckpt
        every_n_epochs=3,
        save_top_k=-1,                      # 不筛选，按周期都保存
        save_on_train_epoch_end=True        # 在每个 train epoch 结束时触发
    )

    logger = CSVLogger('logs', name=config.log_name)

    model = Supervision_Train(config)
    if getattr(config, "pretrained_ckpt_path", None):
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator='gpu',
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[best_ckpt, periodic_ckpt],  # 同时启用两个回调
        logger=logger
    )
    trainer.fit(model=model, ckpt_path=getattr(config, "resume_ckpt_path", None))

if __name__ == "__main__":
    main()
