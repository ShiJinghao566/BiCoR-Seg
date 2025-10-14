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
import pytorch_model_summary
from torchinfo import summary
import torchsummary

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
    arg("-c", "--config_path", type=Path, required=True)
    return parser.parse_args()

def _fdl_sw_over_sb(D: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, N, C = D.shape
    class_means = D.mean(dim=0)
    diff_intra = D - class_means.unsqueeze(0)
    Sw = (diff_intra ** 2).sum(dim=2).mean()
    diff_inter = class_means.unsqueeze(1) - class_means.unsqueeze(0)
    Sb_mat = (diff_inter ** 2).sum(dim=2)
    triu = torch.triu_indices(N, N, offset=1)
    Sb = Sb_mat[triu[0], triu[1]].mean()
    Sb = torch.clamp(Sb, min=1e-4)
    return Sw / (Sb + eps)

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)
        self.has_contrastive_loss = getattr(config, "has_contrastive_loss", False)
        self.lambda_contrastive = getattr(config, "lambda_contrastive", 1.0)
        self.use_heatmap_deep = getattr(config, "use_heatmap_deep", False)
        self.heatmap_beta = getattr(config, "heatmap_beta", [])
        self.lambda_heatmap = getattr(config, "lambda_heatmap", 1.0)
        self.fdl_loss_fn = getattr(config, "fdl_loss", None)
        self.use_fdl_final = getattr(config, "use_fdl_final", False)
        self.use_fdl_multi = getattr(config, "use_fdl_multi", False)
        self.lambda_fdl_final = getattr(config, "lambda_fdl_final", 1.0)
        self.lambda_fdl_multi = getattr(config, "lambda_fdl_multi", 1.0)
        self.fdl_alpha = getattr(config, "fdl_alpha", [])
        self.fdl_eps = getattr(config, "fdl_eps", 1e-6)

    def forward(self, x):
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.net(img)
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
        if aux_logits is not None:
            total_loss = self.loss((seg_logits, aux_logits), mask)
        else:
            total_loss = self.loss(seg_logits, mask)
        if self.has_contrastive_loss and (loss_con is not None):
            total_loss = total_loss + self.lambda_contrastive * loss_con
        if self.use_heatmap_deep and len(H_list) > 0:
            beta = self.heatmap_beta if len(self.heatmap_beta) == len(H_list) else [1.0] * len(H_list)
            loss_heat = 0.0
            for l, h_logits in enumerate(H_list):
                h_up = F.interpolate(h_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                loss_heat = loss_heat + beta[l] * self.loss(h_up, mask)
            total_loss = total_loss + self.lambda_heatmap * loss_heat
        if (self.use_fdl_final or self.use_fdl_multi) and len(D_list) > 0:
            def _call_fdl_fn(D):
                if self.fdl_loss_fn is not None:
                    try:
                        return self.fdl_loss_fn(D)
                    except TypeError:
                        return self.fdl_loss_fn(D, eps=self.fdl_eps)
                else:
                    return _fdl_sw_over_sb(D, eps=self.fdl_eps)
            if self.use_fdl_final:
                loss_fdl_final = _call_fdl_fn(D_list[-1])
                total_loss = total_loss + self.lambda_fdl_final * loss_fdl_final
            if self.use_fdl_multi:
                alpha = self.fdl_alpha if len(self.fdl_alpha) == len(D_list) else [1.0] * len(D_list)
                loss_fdl_m = 0.0
                for l, Dl in enumerate(D_list):
                    loss_fdl_m = loss_fdl_m + alpha[l] * _call_fdl_fn(Dl)
                total_loss = total_loss + self.lambda_fdl_multi * loss_fdl_m
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
        eval_value = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        print('train:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
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
        eval_value = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)
    best_ckpt = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name
    )
    periodic_ckpt = ModelCheckpoint(
        dirpath=config.weights_path,
        filename=f"{config.weights_name}-e{{epoch:04d}}",
        every_n_epochs=3,
        save_top_k=-1,
        save_on_train_epoch_end=True
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
        callbacks=[best_ckpt, periodic_ckpt],
        logger=logger
    )
    trainer.fit(model=model, ckpt_path=getattr(config, "resume_ckpt_path", None))

if __name__ == "__main__":
    main()
