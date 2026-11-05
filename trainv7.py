"""
多任务并行 CNN-LSTM 训练脚本（v6：修复HW分支 forward/初始化以对齐单任务）（完全对齐你两套原脚本的“出图文件名/图形内容/标题”）

并行输出：
- 高度/宽度预测 (2维)
- 颗粒占比预测 (1维)

同一输入：
- X: (B, 30, 8)

对齐的出图（与原脚本一致）：

[高度/宽度脚本 train_cnnlstm_v1.py 对齐]
- loss_height_width.png
- total_loss_curve.png
- mape_height_width.png
- height_true_predicted_line.png
- width_true_predicted_line.png
- height_relative_error.png
- width_relative_error.png

[颗粒占比脚本 kelizhanbi_train_fixed.py 对齐]
- kelizhanbi_loss_curves.png
- kelizhanbi_mse_curves.png
- kelizhanbi_mae_curves.png
- kelizhanbi_particle_prediction_results.png
- kelizhanbi_test_relative_error.png
- particle_true_predicted_line_kelizhanbi.png   (config['file_paths']['true_predicted_line'])

说明：
- 为了“图像内容对齐”，本脚本会在每个epoch结束后统计整套 train/val 的预测（反归一化）以计算MAPE/MSE/MAE等，
  这与两套原脚本的做法一致，但会稍微增加每个epoch的计算量。
"""

import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd

# 关键：使用无GUI后端，避免 plt.show() 阻塞（但仍保留 show 调用以对齐原脚本结构）
import matplotlib
matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from torchvision import models

# 优先使用带 meta 的 v3（提供 test_gids/test_sids，用于 split_mode="global" 下按 group_id+时间顺序画图）
try:
    from data_loader_v7 import MultiTaskBNDataLoader
except Exception:  # pragma: no cover
    try:
        from data_loader_cnnlstm_multitask_parallel_aligned_v3_meta import MultiTaskBNDataLoader
    except Exception:  # pragma: no cover
        try:
            from data_loader_cnnlstm_multitask_parallel_aligned_v2 import MultiTaskBNDataLoader
        except Exception:  # pragma: no cover
            from data_loader_cnnlstm_multitask_parallel_aligned import MultiTaskBNDataLoader

warnings.filterwarnings("ignore")
plt.rcParams["axes.unicode_minus"] = False


TOTAL_LOSS_FACTOR = 0.5
TOTAL_LOSS_FORMULA = "0.5 * (w_hw * loss_hw + w_p * loss_p)"


def combine_task_losses(loss_hw, loss_p, w_hw, w_p):
    """w_hw/w_p are lambda_hw/lambda_p; average the two weighted branches."""
    return TOTAL_LOSS_FACTOR * (w_hw * loss_hw + w_p * loss_p)


# -------------------- metrics (与原脚本口径一致) --------------------
def compute_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    eps = 1e-8
    denom = np.abs(y_true)
    mask = denom > eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (denom[mask] + eps))) * 100.0)


def compute_metrics_particle(y_true, y_pred):
    """对齐 kelizhanbi_train_fixed.py 的 compute_metrics 输出字段"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = -1.0

    # 这里不强制输出mape，原脚本测试集会另外算；训练曲线只画loss/mse/mae
    return {"mse": mse, "mae": mae, "r2_score": r2}


# -------------------- models: standard Conv1D-LSTM multitask architecture --------------------
class SharedCNNLSTMBackbone(nn.Module):
    """Shared standard sequence CNN-LSTM backbone.

    Input shape: (B, T=30, F=8)
    Conv1D stack: (B, 8, 30) -> (B, 32, 30) -> (B, 64, 30)
    LSTM output: (B, 64) from the last time step
    """

    def __init__(
        self,
        input_features=8,
        conv_channels=(32, 64),
        kernel_size=3,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        padding = int(kernel_size) // 2
        c1, c2 = int(conv_channels[0]), int(conv_channels[1])

        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, c1, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.constant_(p, 0.0)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B, T, F), got {tuple(x.shape)}")
        x = x.transpose(1, 2)      # (B, F, T)
        x = self.cnn(x)            # (B, 64, T)
        x = x.transpose(1, 2)      # (B, T, 64)
        out, _ = self.lstm(x)      # (B, T, H)
        return out[:, -1, :]


class RegressionHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(64, 32), dropout=0.3, output_dim=1):
        super().__init__()
        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, features):
        return self.net(features)


class ParallelMultiTaskModel(nn.Module):
    """Shared CNN-LSTM backbone with two task-specific regression heads."""

    def __init__(self, backbone_cfg=None, hw_head_cfg=None, p_head_cfg=None):
        super().__init__()
        base = dict(input_features=8, conv_channels=(32, 64), kernel_size=3, lstm_hidden=64, lstm_layers=2, dropout=0.3)
        base.update(backbone_cfg or {})
        self.backbone = SharedCNNLSTMBackbone(**base)
        head_base = dict(input_dim=int(base["lstm_hidden"]), hidden_dims=(64, 32), dropout=float(base["dropout"]))
        hw_params = dict(head_base)
        hw_params.update(hw_head_cfg or {})
        p_params = dict(head_base)
        p_params.update(p_head_cfg or {})
        self.hw_head = RegressionHead(output_dim=2, **hw_params)
        self.p_head = RegressionHead(output_dim=1, **p_params)

    def forward(self, x):
        features = self.backbone(x)
        return self.hw_head(features), self.p_head(features)


# -------------------- train/eval helpers --------------------
def outpath(outdir, name):
    return os.path.join(outdir, name) if outdir else name



def _make_sequence_samples(group_X, group_y_hw, group_y_p, seq_length, stride=1, include_partial=False):
    """Build sequence samples without converting them to statistical images."""
    X_seqs, y_hw_seqs, y_p_seqs, sids = [], [], [], []
    group_len = len(group_X)
    seq_length = int(seq_length)
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    if include_partial:
        for i in range(0, group_len, stride):
            start = max(0, i - seq_length + 1)
            seq = group_X[start:i + 1]
            if len(seq) < seq_length:
                pad_value = np.mean(seq, axis=0)
                padding = np.tile(pad_value, (seq_length - len(seq), 1))
                seq = np.vstack([padding, seq])
            X_seqs.append(seq)
            y_hw_seqs.append(group_y_hw[i])
            y_p_seqs.append(group_y_p[i])
            sids.append(i)
    else:
        if group_len < seq_length:
            return None, None, None, None
        for start in range(0, group_len - seq_length + 1, stride):
            end = start + seq_length
            X_seqs.append(group_X[start:end])
            y_hw_seqs.append(group_y_hw[end - 1])
            y_p_seqs.append(group_y_p[end - 1])
            sids.append(end - 1)

    return (
        np.asarray(X_seqs, dtype=np.float32),
        np.asarray(y_hw_seqs, dtype=np.float32),
        np.asarray(y_p_seqs, dtype=np.float32),
        np.asarray(sids, dtype=np.int32),
    )


def _loader_from_arrays(X, y_hw, y_p, batch_size, shuffle, num_workers=0, pin_memory=True):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_hw, dtype=torch.float32),
        torch.tensor(y_p, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )


def create_sequence_data_loaders(dl: MultiTaskBNDataLoader, batch_size=32, num_workers=0, pin_memory=True, split_mode="global"):
    """Create standard CNN-LSTM loaders with input shape (B, seq_length, num_features)."""
    X_raw, y_hw_raw, y_p_raw, _ = dl.load_and_clean_data()
    X_scaled_all, _ = dl.preprocess_features(X_raw, global_fit=True)
    y_hw_scaled_all, _ = dl.preprocess_targets_hw(y_hw_raw, np.ones(len(y_hw_raw), dtype=bool), global_fit=True)
    y_p_scaled_all, _ = dl.preprocess_targets_p(y_p_raw, np.ones(len(y_p_raw), dtype=bool), global_fit=True)

    total_rows = len(X_raw)
    group_size = int(dl.group_size)
    seq_length = int(dl.seq_length)
    stride = int(getattr(dl, "stride", 1))
    num_groups = int(np.ceil(total_rows / group_size))

    all_X, all_y_hw, all_y_p, all_gid, all_sid = [], [], [], [], []
    per_group = []
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, total_rows)
        Xg, yhg, ypg, sid = _make_sequence_samples(
            X_scaled_all[start:end],
            y_hw_scaled_all[start:end],
            y_p_scaled_all[start:end],
            seq_length=seq_length,
            stride=stride,
            include_partial=False,
        )
        if Xg is None or len(Xg) == 0:
            continue
        gid = np.full((len(Xg),), g, dtype=np.int32)
        all_X.append(Xg)
        all_y_hw.append(yhg)
        all_y_p.append(ypg)
        all_gid.append(gid)
        all_sid.append(sid)
        per_group.append((Xg, yhg, ypg, gid, sid))

    if not all_X:
        raise ValueError("Not enough data to build sequence dataset; check group_size, seq_length, and row count")

    X_all = np.concatenate(all_X, axis=0)
    y_hw_all = np.concatenate(all_y_hw, axis=0)
    y_p_all = np.concatenate(all_y_p, axis=0)
    gid_all = np.concatenate(all_gid, axis=0)
    sid_all = np.concatenate(all_sid, axis=0)

    split_mode = "global" if split_mode is None else str(split_mode).lower().strip()
    if split_mode in {"global", "overall", "all"}:
        X_train, X_test, yhw_train, yhw_test, yp_train, yp_test, gid_train, gid_test, sid_train, sid_test = train_test_split(
            X_all, y_hw_all, y_p_all, gid_all, sid_all,
            test_size=float(dl.test_size),
            random_state=int(dl.random_state),
            shuffle=True,
        )
        val_ratio = float(dl.val_size) / max(1e-8, 1.0 - float(dl.test_size))
        X_train, X_val, yhw_train, yhw_val, yp_train, yp_val, gid_train, gid_val, sid_train, sid_val = train_test_split(
            X_train, yhw_train, yp_train, gid_train, sid_train,
            test_size=val_ratio,
            random_state=int(dl.random_state),
            shuffle=True,
        )
    elif split_mode in {"by_group", "group"}:
        train_parts, val_parts, test_parts = [], [], []
        for Xg, yhg, ypg, gid, sid in per_group:
            if len(Xg) < 3:
                continue
            X_tr, X_te, yhw_tr, yhw_te, yp_tr, yp_te, gid_tr, gid_te, sid_tr, sid_te = train_test_split(
                Xg, yhg, ypg, gid, sid,
                test_size=float(dl.test_size),
                random_state=int(dl.random_state),
                shuffle=True,
            )
            val_ratio = float(dl.val_size) / max(1e-8, 1.0 - float(dl.test_size))
            X_tr, X_va, yhw_tr, yhw_va, yp_tr, yp_va, gid_tr, gid_va, sid_tr, sid_va = train_test_split(
                X_tr, yhw_tr, yp_tr, gid_tr, sid_tr,
                test_size=val_ratio,
                random_state=int(dl.random_state),
                shuffle=True,
            )
            train_parts.append((X_tr, yhw_tr, yp_tr, gid_tr, sid_tr))
            val_parts.append((X_va, yhw_va, yp_va, gid_va, sid_va))
            test_parts.append((X_te, yhw_te, yp_te, gid_te, sid_te))

        def _cat(parts, pos):
            return np.concatenate([p[pos] for p in parts], axis=0)
        X_train, yhw_train, yp_train, gid_train, sid_train = (_cat(train_parts, i) for i in range(5))
        X_val, yhw_val, yp_val, gid_val, sid_val = (_cat(val_parts, i) for i in range(5))
        X_test, yhw_test, yp_test, gid_test, sid_test = (_cat(test_parts, i) for i in range(5))
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    dl.train_gids, dl.val_gids, dl.test_gids = gid_train.astype(np.int32), gid_val.astype(np.int32), gid_test.astype(np.int32)
    dl.train_sids, dl.val_sids, dl.test_sids = sid_train.astype(np.int32), sid_val.astype(np.int32), sid_test.astype(np.int32)
    dl.all_test_images, dl.all_test_targets_hw, dl.all_test_targets_p = [], [], []
    for g in sorted(set(gid_test.tolist())):
        idx = gid_test == g
        order = np.argsort(sid_test[idx], kind="stable")
        dl.all_test_images.append(X_test[idx][order].astype(np.float32))
        dl.all_test_targets_hw.append(yhw_test[idx][order].astype(np.float32))
        dl.all_test_targets_p.append(yp_test[idx][order].astype(np.float32))

    dl.data_stats.update({
        "total_rows": int(total_rows),
        "num_groups": int(num_groups),
        "group_size_rows": int(group_size),
        "total_sequences": int(len(X_all)),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "seq_length": int(seq_length),
        "stride": int(stride),
        "num_features": int(X_all.shape[-1]),
        "input_shape": "(batch, seq_length, num_features)",
        "split_mode": split_mode,
    })

    return (
        _loader_from_arrays(X_train, yhw_train, yp_train, batch_size, True, num_workers, pin_memory),
        _loader_from_arrays(X_val, yhw_val, yp_val, batch_size, False, num_workers, pin_memory),
        _loader_from_arrays(X_test, yhw_test, yp_test, batch_size, False, num_workers, pin_memory),
    )


def build_all_loader(dl: MultiTaskBNDataLoader, batch_size: int, num_workers: int = 0, pin_memory: bool = True):
    """Build a full-row prediction loader with standard sequence input."""
    X_raw, y_hw_raw, y_p_raw, _ = dl.load_and_clean_data()
    X_scaled_all, _ = dl.preprocess_features(X_raw, global_fit=False)
    y_hw_scaled_all, _ = dl.preprocess_targets_hw(y_hw_raw, np.ones(len(y_hw_raw), dtype=bool), global_fit=False)
    y_p_scaled_all, _ = dl.preprocess_targets_p(y_p_raw, np.ones(len(y_p_raw), dtype=bool), global_fit=False)

    total_rows = len(X_raw)
    group_size = int(dl.group_size)
    seq_length = int(dl.seq_length)
    stride = int(getattr(dl, "stride", 1))
    num_groups = int(np.ceil(total_rows / group_size))

    all_X, all_y_hw, all_y_p, all_gid, all_sid = [], [], [], [], []
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, total_rows)
        Xg, yhg, ypg, sid = _make_sequence_samples(
            X_scaled_all[start:end],
            y_hw_scaled_all[start:end],
            y_p_scaled_all[start:end],
            seq_length=seq_length,
            stride=stride,
            include_partial=False,
        )
        if Xg is None or len(Xg) == 0:
            continue
        all_X.append(Xg)
        all_y_hw.append(yhg)
        all_y_p.append(ypg)
        all_gid.append(np.full((len(Xg),), g, dtype=np.int32))
        all_sid.append(sid)

    if not all_X:
        raise ValueError("Unable to build full sequence dataset; check group_size, seq_length, and row count")

    X_all = np.concatenate(all_X, axis=0)
    y_hw_all = np.concatenate(all_y_hw, axis=0)
    y_p_all = np.concatenate(all_y_p, axis=0)
    gid_all = np.concatenate(all_gid, axis=0)
    sid_all = np.concatenate(all_sid, axis=0)
    return _loader_from_arrays(X_all, y_hw_all, y_p_all, batch_size, False, num_workers, pin_memory), gid_all, sid_all


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler,
    hw_loss_weight,
    p_loss_params,
    w_hw,
    w_p,
    amp,
    grad_accum_steps=1,
    clip_hw=1.0,
    clip_p=3.0,
):
    """单个 epoch 的训练/验证。

    - 训练时：支持两个 optimizer（高度宽度 / 颗粒占比）各自 step。
    - 支持梯度累积（grad_accum_steps），以在 batch 固定时模拟更大的有效 batch。
    - 支持分别梯度裁剪（clip_hw / clip_p），对齐两套单任务脚本的稳定性策略。

    注意：loss/指标的计算逻辑保持与原脚本一致。
    """

    is_train = optimizer is not None
    model.train(is_train)

    grad_accum_steps = int(max(1, grad_accum_steps))

    # sums
    total_loss = 0.0
    hw_loss_total = 0.0
    hw_loss_h = 0.0
    hw_loss_w = 0.0
    p_loss = 0.0
    n_samples = 0  # total number of samples (weighted mean)

    # preds/targets（用于epoch级指标计算与曲线对齐）
    hw_preds, hw_tgts = [], []
    p_preds, p_tgts = [], []

    for batch_idx, (x, y_hw, y_p) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y_hw = y_hw.to(device, non_blocking=True)
        y_p = y_p.to(device, non_blocking=True)

        bs = int(x.shape[0])
        n_samples += bs

        # 梯度清零：仅在一个累积窗口开始时清零
        if is_train and (batch_idx % grad_accum_steps == 0):
            optimizer.zero_grad(set_to_none=True)
        # Forward: HW 分支使用 AMP（若启用），P 分支强制 FP32（稳定 BN running stats / 对齐单任务）
        with autocast(enabled=amp):
            pred_hw, pred_p = model(x)

        # Loss: 统一在 FP32 里算，避免半精度 MSE/MAE 导致不稳定
        pred_hw_f = pred_hw.float()
        pred_p_f = pred_p.float()
        y_hw_f = y_hw.float()
        y_p_f = y_p.float()

        # ------------------ 高度/宽度：保持原脚本的 loss 设计 ------------------
        l_h = F.mse_loss(pred_hw_f[:, 0], y_hw_f[:, 0])  # height MSE
        l_w = F.mse_loss(pred_hw_f[:, 1], y_hw_f[:, 1])  # width  MSE
        l_hw = hw_loss_weight * l_h + (1.0 - hw_loss_weight) * l_w

        # ------------------ 颗粒占比：保持原脚本的 loss 设计 ------------------
        beta = float(p_loss_params.get("huber_beta", 0.1))
        weights = p_loss_params.get("loss_weights", {"huber": 0.4, "mse": 0.4, "mae": 0.2})
        w_hu = float(weights.get("huber", 0.4))
        w_mse = float(weights.get("mse", 0.4))
        w_mae = float(weights.get("mae", 0.2))

        huber = F.smooth_l1_loss(pred_p_f, y_p_f, beta=beta)
        mse = F.mse_loss(pred_p_f, y_p_f)
        mae = F.l1_loss(pred_p_f, y_p_f)
        l_p = w_hu * huber + w_mse * mse + w_mae * mae

        # 多任务总损失：0.5 * (lambda_hw * L_hw + lambda_p * L_p)
        loss = combine_task_losses(l_hw, l_p, w_hw, w_p)

        # 梯度累积：缩放 loss，保证与不累积时梯度期望一致
        loss_for_bp = loss / float(grad_accum_steps)

        if is_train:
            if scaler is not None and amp:
                scaler.scale(loss_for_bp).backward()
            else:
                loss_for_bp.backward()

            # 到达累积窗口末尾 / 最后一个 batch：执行 step
            is_accum_end = ((batch_idx + 1) % grad_accum_steps == 0)
            is_last = (batch_idx + 1 == len(loader))
            if is_accum_end or is_last:
                if scaler is not None and amp:
                    # unscale 后再做梯度裁剪
                    scaler.unscale_(optimizer)
                    clip_candidates = [v for v in (clip_hw, clip_p) if v is not None and v > 0]
                    if clip_candidates:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max(clip_candidates)))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_candidates = [v for v in (clip_hw, clip_p) if v is not None and v > 0]
                    if clip_candidates:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max(clip_candidates)))
                    optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * bs
        hw_loss_total += float(l_hw.detach().cpu().item()) * bs
        p_loss += float(l_p.detach().cpu().item()) * bs
        hw_loss_h += float(l_h.detach().cpu().item()) * bs
        hw_loss_w += float(l_w.detach().cpu().item()) * bs

        hw_preds.append(pred_hw.detach().cpu().numpy())
        hw_tgts.append(y_hw.detach().cpu().numpy())
        p_preds.append(pred_p_f.detach().cpu().numpy())
        p_tgts.append(y_p.detach().cpu().numpy())

    if n_samples == 0:
        raise RuntimeError("DataLoader为空：没有任何样本")

    hw_preds = np.concatenate(hw_preds, axis=0)
    hw_tgts = np.concatenate(hw_tgts, axis=0)
    p_preds = np.concatenate(p_preds, axis=0)
    p_tgts = np.concatenate(p_tgts, axis=0)

    return {
        "loss_total": total_loss / n_samples,
        "loss_hw": hw_loss_total / n_samples,
        "loss_hw_h": hw_loss_h / n_samples,
        "loss_hw_w": hw_loss_w / n_samples,
        "loss_p": p_loss / n_samples,
        "pred_hw_scaled": hw_preds,
        "tgt_hw_scaled": hw_tgts,
        "pred_p_scaled": p_preds,
        "tgt_p_scaled": p_tgts,
    }


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    phw, pp, thw, tp = [], [], [], []
    for x, y_hw, y_p in loader:
        x = x.to(device, non_blocking=True)
        with autocast(enabled=amp):
            yhw_hat, yp_hat = model(x)
        phw.append(yhw_hat.detach().cpu().numpy())
        pp.append(yp_hat.float().detach().cpu().numpy())
        thw.append(y_hw.numpy())
        tp.append(y_p.numpy())
    return np.concatenate(phw), np.concatenate(pp), np.concatenate(thw), np.concatenate(tp)


# -------------------- plotting (严格对齐原图) --------------------
def plot_total_loss(history_total, outdir):
    """绘制与优化目标一致的总损失：0.5 * (w_hw * loss_hw + w_p * loss_p)。"""
    plt.figure(figsize=(10, 6))
    plt.plot(history_total["epoch"], history_total["train_loss"], label="Training Loss", color="blue")
    plt.plot(history_total["epoch"], history_total["val_loss"], label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss (Height/Width + Particle)")
    plt.title("Total Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "total_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_hw_all(
    history_hw,
    test_targets_denorm,
    test_predictions_denorm,
    height_r2,
    width_r2,
    test_mape_height,
    test_mape_width,
    outdir,
    sorted_gids=None,
):
    # 1) loss_height_width.png
    plt.figure(figsize=(10, 6))
    plt.plot(history_hw["epoch"], history_hw["train_loss_height"], label="Height Training Loss", color="blue")
    plt.plot(history_hw["epoch"], history_hw["val_loss_height"], label="Height Validation Loss", color="red")
    plt.plot(history_hw["epoch"], history_hw["train_loss_width"], label="Width Training Loss", color="green")
    plt.plot(history_hw["epoch"], history_hw["val_loss_width"], label="Width Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Height and Width Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "loss_height_width.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 3) mape_height_width.png
    plt.figure(figsize=(10, 6))
    plt.plot(history_hw["epoch"], history_hw["train_mape_height"], label="Height Training MAPE", color="blue")
    plt.plot(history_hw["epoch"], history_hw["val_mape_height"], label="Height Validation MAPE", color="red")
    plt.plot(history_hw["epoch"], history_hw["train_mape_width"], label="Width Training MAPE", color="green")
    plt.plot(history_hw["epoch"], history_hw["val_mape_width"], label="Width Validation MAPE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE (%)")
    plt.title("Height and Width MAPE Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "mape_height_width.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 4) height_true_predicted_line.png

    plt.figure(figsize=(12, 6))

    # 这里不再按数值排序；要求：严格按 group_id 的真实工况顺序（组内按时间序列顺序）
    # 因此：调用该函数前，应已将 test_targets/preds 重新排序好。
    sample_indices = np.arange(len(test_targets_denorm))

    # True（阶梯线）
    plt.step(sample_indices, test_targets_denorm[:, 0], where="post", linewidth=2.5, label="True Height")
    # Pred（虚线）
    plt.plot(sample_indices, test_predictions_denorm[:, 0], "r--", linewidth=2, label="Predicted Height")

    # 画出组间分割线（可选）
    if sorted_gids is not None and len(sorted_gids) == len(sample_indices):
        boundaries = np.where(np.diff(sorted_gids) != 0)[0] + 1
        for b in boundaries:
            plt.axvline(b - 0.5, linewidth=0.8, alpha=0.25)

    plt.xlabel("Sample Index")
    plt.ylabel("Height")
    plt.title(f"Height True vs Predicted Values (R² = {height_r2:.3f}, MAPE = {test_mape_height:.2f}%)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "height_true_predicted_line.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 5) width_true_predicted_line.png

    plt.figure(figsize=(12, 6))

    sample_indices = np.arange(len(test_targets_denorm))

    # True（阶梯线）
    plt.step(sample_indices, test_targets_denorm[:, 1], where="post", linewidth=2.5, label="True Width")
    # Pred（虚线）
    plt.plot(sample_indices, test_predictions_denorm[:, 1], "r--", linewidth=2, label="Predicted Width")

    # 组间分割线（可选）
    if sorted_gids is not None and len(sorted_gids) == len(sample_indices):
        boundaries = np.where(np.diff(sorted_gids) != 0)[0] + 1
        for b in boundaries:
            plt.axvline(b - 0.5, linewidth=0.8, alpha=0.25)

    plt.xlabel("Sample Index")
    plt.ylabel("Width")
    plt.title(f"Width True vs Predicted Values (R² = {width_r2:.3f}, MAPE = {test_mape_width:.2f}%)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "width_true_predicted_line.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 6) height_relative_error.png
    # 6) height_relative_error.png
    sample_indices = np.arange(len(test_targets_denorm))

    plt.figure(figsize=(12, 6))
    height_relative_error = (test_predictions_denorm[:, 0] - test_targets_denorm[:, 0]) / test_targets_denorm[:, 0]
    plt.bar(sample_indices, height_relative_error, color="blue", label="Height Relative Error")
    plt.xlabel("Sample Index")
    plt.ylabel("Relative Error")
    plt.title(f"Height Relative Error (Mean: {height_relative_error.mean():.4f}, Std: {height_relative_error.std():.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "height_relative_error.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 7) width_relative_error.png
    plt.figure(figsize=(12, 6))
    width_relative_error = (test_predictions_denorm[:, 1] - test_targets_denorm[:, 1]) / test_targets_denorm[:, 1]
    plt.bar(sample_indices, width_relative_error, color="green", label="Width Relative Error")
    plt.xlabel("Sample Index")
    plt.ylabel("Relative Error")
    plt.title(f"Width Relative Error (Mean: {width_relative_error.mean():.4f}, Std: {width_relative_error.std():.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "width_relative_error.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_particle_all(history_p, test_targets_denorm_all, test_predictions_denorm_all, all_group_targets, all_group_preds, outdir):
    # 1) kelizhanbi_loss_curves.png
    plt.figure(figsize=(10, 6))
    plt.plot(history_p["epoch"], history_p["train_loss"], label="Training Loss")
    plt.plot(history_p["epoch"], history_p["val_loss"], label="Validation Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "kelizhanbi_loss_curves.png"))
    print("📊 损失曲线已保存到 kelizhanbi_loss_curves.png")
    plt.close()

    # 2) kelizhanbi_mse_curves.png
    plt.figure(figsize=(10, 6))
    plt.plot(history_p["epoch"], history_p["train_mse"], label="Training MSE")
    plt.plot(history_p["epoch"], history_p["val_mse"], label="Validation MSE")
    plt.title("MSE Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "kelizhanbi_mse_curves.png"))
    print("📊 MSE曲线已保存到 kelizhanbi_mse_curves.png")
    plt.close()

    # 3) kelizhanbi_mae_curves.png
    plt.figure(figsize=(10, 6))
    plt.plot(history_p["epoch"], history_p["train_mae"], label="Training MAE")
    plt.plot(history_p["epoch"], history_p["val_mae"], label="Validation MAE")
    plt.title("MAE Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath(outdir, "kelizhanbi_mae_curves.png"))
    print("📊 MAE曲线已保存到 kelizhanbi_mae_curves.png")
    plt.close()

    # 4) kelizhanbi_particle_prediction_results.png（散点）
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(test_targets_denorm_all, test_predictions_denorm_all, alpha=0.5)
        plt.plot(
            [test_targets_denorm_all.min(), test_targets_denorm_all.max()],
            [test_targets_denorm_all.min(), test_targets_denorm_all.max()],
            "r--",
            lw=2,
        )
        plt.title("Particle Proportion Prediction Results")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        plt.savefig(outpath(outdir, "kelizhanbi_particle_prediction_results.png"))
        print("📊 预测结果散点图已保存到 kelizhanbi_particle_prediction_results.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  绘制预测结果散点图时出错: {e}")

    # 5) kelizhanbi_test_relative_error.png（柱状）

    # 5) kelizhanbi_test_relative_error.png（柱状）
    # 说明：你看到的“尖峰”通常来自某个样本的 |(pred-true)/true| 很大：
    # - true 值较小（颗粒占比本来就在 0.02~0.08），即使绝对误差不大，相对误差也会被放大
    # - 或者某个样本/组是明显离群点
    try:
        plt.figure(figsize=(15, 8))
        eps = 1e-8

        # 优先按“逐组拼接”的顺序来画（更符合‘按组’这个标题）
        if len(all_group_targets) > 0 and len(all_group_preds) > 0:
            all_t = np.concatenate(all_group_targets).flatten()
            all_p = np.concatenate(all_group_preds).flatten()

            # 同时构建每个点对应的 group 索引，方便定位尖峰来自哪一组
            g_index = []
            for gi, tg in enumerate(all_group_targets):
                g_index.extend([gi] * len(tg))
            g_index = np.array(g_index, dtype=int)
        else:
            all_t = test_targets_denorm_all.flatten()
            all_p = test_predictions_denorm_all.flatten()
            g_index = None

        relative_error = (all_p - all_t) / (all_t + eps)

        # 定位最大尖峰（按绝对值）
        max_i = int(np.argmax(np.abs(relative_error))) if len(relative_error) > 0 else -1
        if max_i >= 0:
            if g_index is None:
                print(f"[Particle] Max |rel_error|={relative_error[max_i]:.4f} at idx={max_i} | true={all_t[max_i]:.6f} pred={all_p[max_i]:.6f}")
            else:
                print(f"[Particle] Max |rel_error|={relative_error[max_i]:.4f} at idx={max_i} (group={g_index[max_i]}) | true={all_t[max_i]:.6f} pred={all_p[max_i]:.6f}")

        plt.bar(range(len(relative_error)), relative_error, color="blue", alpha=0.7)

        plt.title(f"Particle Relative Error (Mean: {relative_error.mean():.4f}, Std: {relative_error.std():.4f})")
        plt.xlabel("Sample Index (Concatenated by Group)")
        plt.ylabel("Relative Error")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(["Particle Relative Error"], loc="upper right")

        plt.tight_layout()
        plt.savefig(outpath(outdir, "kelizhanbi_test_relative_error.png"), dpi=300, bbox_inches="tight")
        print("Relative error plot saved to kelizhanbi_test_relative_error.png")
        plt.close()
    except Exception as e:
        print(f"Plot particle relative error failed: {e}")

    # 6) particle_true_predicted_line_kelizhanbi.png（折线，合并所有组）
    # 6) particle_true_predicted_line_kelizhanbi.png（折线，合并所有组）
    if len(all_group_targets) > 0 and len(all_group_preds) > 0:
        try:
            plt.figure(figsize=(14, 6))
            all_test_targets = np.concatenate(all_group_targets).flatten()
            all_test_predictions = np.concatenate(all_group_preds).flatten()

            overall_r2 = r2_score(all_test_targets, all_test_predictions) if len(all_test_targets) > 2 else -1.0
            overall_mape = compute_mape(all_test_targets, all_test_predictions)

            # True（实线）
            plt.plot(
                np.arange(len(all_test_targets)),
                all_test_targets,
                linewidth=2.5,
                color="#1f77b4",
                label="True Values",
            )
            # Pred（虚线）
            plt.plot(
                np.arange(len(all_test_predictions)),
                all_test_predictions,
                "r--",
                linewidth=2,
                label="Predicted Values",
            )

            plt.xlabel("Sample Index")
            plt.ylabel("Particle Proportion")
            plt.title(f"Particle Proportion True vs Predicted Values ($R^2$ = {overall_r2:.3f}, MAPE = {overall_mape:.2f}%)")
            plt.grid(True)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(outpath(outdir, "particle_true_predicted_line_kelizhanbi.png"), dpi=300, bbox_inches="tight")
            print("📊 颗粒占比预测值与真实值对比折线图已保存到 particle_true_predicted_line_kelizhanbi.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  绘制颗粒占比预测值与真实值对比折线图时出错: {e}")


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="师兄数据.xlsx")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq_length", type=int, default=30, help="滑动窗口大小")
    ap.add_argument("--stride", type=int, default=30, help="滑动步长")
    ap.add_argument("--group_size", type=int, default=1800, help="每组原始数据行数")
    ap.add_argument("--split_mode", type=str, default="chronological",
                    choices=("chronological", "global", "by_group"),
                    help=("数据集划分方式：chronological(默认，每组按时间顺序"
                          "8:1:1划分训练/测试/验证目标时刻，窗口仅在组内滑动) / "
                          "global(全局随机切分窗口) / by_group(组内随机切分窗口)"))
    ap.add_argument("--lr", type=float, default=None,
                    help="(兼容旧参数) 若提供则同时覆盖 lr_hw 与 lr_p")
    ap.add_argument("--lr_hw", type=float, default=1e-4,
                    help="高度/宽度分支学习率（对齐单任务最优）")
    ap.add_argument("--lr_p", type=float, default=1e-4,
                    help="颗粒占比分支学习率（对齐单任务最优）")
    ap.add_argument("--wd_hw", type=float, default=1e-4,
                    help="高度/宽度分支 weight_decay（对齐单任务脚本 best_params['weight_decay']）")
    ap.add_argument("--wd_p", type=float, default=5e-5,
                    help="颗粒占比分支 weight_decay（对齐单任务 config['optimizer_params']['weight_decay']）")
    ap.add_argument("--sched_factor", type=float, default=None,
                    help="(旧参数) 若提供则同时覆盖 hw/p 的调度器 factor；不提供则使用各自默认")
    ap.add_argument("--sched_patience", type=int, default=None,
                    help="(旧参数) 若提供则同时覆盖 hw/p 的调度器 patience；不提供则使用各自默认")
    ap.add_argument("--min_lr_hw", type=float, default=1e-8,
                    help="高度/宽度分支最小学习率（ReduceLROnPlateau 的 min_lr[0]）")
    ap.add_argument("--min_lr_p", type=float, default=1e-6,
                    help="颗粒占比分支最小学习率（ReduceLROnPlateau 的 min_lr[1]）")
    # 任务各自的调度器参数（默认对齐两套单任务脚本）
    ap.add_argument("--hw_sched_factor", type=float, default=0.1,
                    help="高度/宽度分支 ReduceLROnPlateau factor（对齐单任务 best_params['scheduler_factor']）")
    ap.add_argument("--hw_sched_patience", type=int, default=5,
                    help="高度/宽度分支 ReduceLROnPlateau patience（对齐单任务 best_params['scheduler_patience']）")
    ap.add_argument("--p_sched_factor", type=float, default=0.5,
                    help="颗粒占比分支 ReduceLROnPlateau factor（对齐颗粒脚本 config['scheduler_params']['factor']）")
    ap.add_argument("--p_sched_patience", type=int, default=15,
                    help="颗粒占比分支 ReduceLROnPlateau patience（对齐颗粒脚本 config['scheduler_params']['patience']）")

    # 训练稳定性参数（对齐两套单任务脚本的梯度裁剪策略）
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                    help="梯度累积步数：>1 可在 batch 不变时模拟更大有效 batch")
    ap.add_argument("--clip_hw", type=float, default=1.0,
                    help="高度/宽度分支梯度裁剪 max_norm（对齐单任务脚本 1.0）")
    ap.add_argument("--clip_p", type=float, default=3.0,
                    help="颗粒占比分支梯度裁剪 max_norm（对齐颗粒脚本 3.0）")
    ap.add_argument("--w_hw", type=float, default=1.0)
    ap.add_argument("--w_p", type=float, default=1.0)
    ap.add_argument("--hw_loss_weight", type=float, default=0.5,
                    help="高度/宽度任务：height MSE 与 width MSE 的加权系数（对应原脚本 best_params['loss_weight']）")
    ap.add_argument("--p_huber_beta", type=float, default=0.1,
                    help="颗粒占比任务：SmoothL1(Huber) 的 beta（对应原脚本 config['loss_params']['huber_beta']）")
    ap.add_argument("--p_w_huber", type=float, default=0.4, help="颗粒占比任务：Huber 权重")
    ap.add_argument("--p_w_mse", type=float, default=0.4, help="颗粒占比任务：MSE 权重")
    ap.add_argument("--p_w_mae", type=float, default=0.2, help="颗粒占比任务：MAE 权重")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--pred_csv", type=str, default="all_predictions.csv", help="训练结束后导出的全量预测CSV文件名（仅三列：height,width,particle）")
    ap.add_argument("--pred_ckpt", type=str, default="best", choices=["best","best_hw","best_p"], help="导出预测CSV时使用的checkpoint：best=best_total, best_hw 或 best_p")
    args = ap.parse_args()
    # Persist the exact objective in future checkpoints; not a tunable weight.
    args.total_loss_factor = TOTAL_LOSS_FACTOR
    args.total_loss_formula = TOTAL_LOSS_FORMULA

    # -------------------- 兼容：输入为目录/通配符的多个 *.csv 时，先合并为临时 xlsx --------------------
    # data_loader 默认读取 xlsx；如果你传入的是目录或通配符，且里面是“成型.csv”一类文件，可自动合并。
    if isinstance(args.data, str):
        data_path = args.data
        is_dir = os.path.isdir(data_path)
        has_glob = any(ch in data_path for ch in ["*", "?", "["])
        is_csv = data_path.lower().endswith(".csv")

        if is_dir or has_glob or is_csv:
            import glob
            csv_files = []
            if is_dir:
                csv_files = sorted(glob.glob(os.path.join(data_path, "*成型.csv"))) or sorted(glob.glob(os.path.join(data_path, "*.csv")))
            elif has_glob:
                csv_files = sorted(glob.glob(data_path))
            elif is_csv:
                csv_files = [data_path]

            if len(csv_files) > 0:
                dfs = []
                for i, fp in enumerate(csv_files):
                    try:
                        df_i = pd.read_csv(fp, header=None)
                        # 多文件拼接时，去掉后续文件的表头行（假设第0行是表头）
                        if i > 0 and len(df_i) > 0:
                            df_i = df_i.iloc[1:, :]
                        dfs.append(df_i)
                    except Exception as _:
                        continue
                if len(dfs) > 0:
                    df_all = pd.concat(dfs, axis=0, ignore_index=True)
                    tmp_xlsx = os.path.join(args.outdir, "_merged_all_xxx.xlsx")
                    df_all.to_excel(tmp_xlsx, header=False, index=False)
                    print(f"ℹ️ 已将 {len(dfs)} 个CSV合并为临时xlsx: {os.path.abspath(tmp_xlsx)}")
                    args.data = tmp_xlsx

    # 兼容旧参数：如果用户只传了 --lr，则两分支共用同一学习率
    if args.lr is not None:
        args.lr_hw = float(args.lr)
        args.lr_p = float(args.lr)

    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and torch.cuda.is_available()

    # data
    dl = MultiTaskBNDataLoader(
        file_path=args.data,
        seq_length=args.seq_length,
        stride=args.stride,
        image_size=(224, 224),
        test_size=0.1,
        val_size=0.1,
        group_size=args.group_size,
        random_state=args.seed,
    )
    train_loader, val_loader, test_loader = dl.create_data_loaders(batch_size=args.batch, num_workers=0, pin_memory=True, split_mode=args.split_mode)

    # model: shared Conv1D-LSTM backbone + two task-specific regression heads
    model = ParallelMultiTaskModel(
        backbone_cfg={},
        hw_head_cfg={},
        p_head_cfg={},
    ).to(device)
    # loss params（保持原两个训练脚本的 loss 设计）
    hw_loss_weight = float(args.hw_loss_weight)

    p_loss_params = {
        "huber_beta": float(args.p_huber_beta),
        "loss_weights": {
            "huber": float(args.p_w_huber),
            "mse": float(args.p_w_mse),
            "mae": float(args.p_w_mae),
        },
    }
    # 若权重和不为1：做一次归一化（保持相对比例），避免用户误输入导致梯度尺度异常
    _sw = sum(p_loss_params["loss_weights"].values())
    if _sw > 0:
        for _k in p_loss_params["loss_weights"]:
            p_loss_params["loss_weights"][_k] /= _sw

    # 使用“分支不同学习率”的 Adam 优化器
    # 注意：两任务的参数完全独立（并行架构），但学习率调度也必须解耦，否则会出现：
    # - 颗粒任务先收敛/波动导致 scheduler 过早降低高度宽度的 lr（高度宽度会明显变差）
    opt = optim.Adam(
        [
            {
                "params": model.backbone.parameters(),
                "lr": float(args.lr_hw),
                "weight_decay": float(args.wd_hw),
                "name": "shared_backbone",
            },
            {
                "params": model.hw_head.parameters(),
                "lr": float(args.lr_hw),
                "weight_decay": float(args.wd_hw),
                "name": "hw_head",
            },
            {
                "params": model.p_head.parameters(),
                "lr": float(args.lr_p),
                "weight_decay": float(args.wd_p),
                "name": "p_head",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 兼容旧参数：若用户设置了 --sched_factor/--sched_patience，则同时覆盖两个任务
    hw_factor = float(args.sched_factor) if args.sched_factor is not None else float(args.hw_sched_factor)
    hw_pat = int(args.sched_patience) if args.sched_patience is not None else int(args.hw_sched_patience)
    p_factor = float(args.sched_factor) if args.sched_factor is not None else float(args.p_sched_factor)
    p_pat = int(args.sched_patience) if args.sched_patience is not None else int(args.p_sched_patience)

    sched_factor = float(args.sched_factor) if args.sched_factor is not None else p_factor
    sched_patience = int(args.sched_patience) if args.sched_patience is not None else p_pat
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=sched_factor,
        patience=sched_patience,
        min_lr=[float(args.min_lr_hw), float(args.min_lr_hw), float(args.min_lr_p)],
    )

    scaler = GradScaler(enabled=amp)


    # histories（分别对齐两套脚本字段）
    history_hw = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_height": [],
        "train_loss_width": [],
        "val_loss_height": [],
        "val_loss_width": [],
        "train_mape_height": [],
        "train_mape_width": [],
        "val_mape_height": [],
        "val_mape_width": [],
        "learning_rate": [],
        "epoch": [],
    }
    history_p = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": [],
        "learning_rate": [],
        "epoch": [],
    }

    history_total = {
        "total_loss_factor": TOTAL_LOSS_FACTOR,
        "total_loss_formula": TOTAL_LOSS_FORMULA,
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "w_hw": float(args.w_hw),
        "w_p": float(args.w_p),
    }

    best_val = float("inf")
    best_hw = float("inf")
    best_p = float("inf")
    best_path = outpath(args.outdir, "best_multitask_model.pt")
    best_hw_path = outpath(args.outdir, "best_hw_model.pt")
    best_p_path = outpath(args.outdir, "best_p_model.pt")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, device,
            optimizer=opt, scaler=scaler,
            hw_loss_weight=hw_loss_weight, p_loss_params=p_loss_params,
            w_hw=args.w_hw, w_p=args.w_p,
            amp=amp,
            grad_accum_steps=args.grad_accum_steps,
            clip_hw=args.clip_hw,
            clip_p=args.clip_p,
        )
        va = run_epoch(
            model, val_loader, device,
            optimizer=None, scaler=None,
            hw_loss_weight=hw_loss_weight, p_loss_params=p_loss_params,
            w_hw=args.w_hw, w_p=args.w_p,
            amp=amp,
        )

        # 对齐高度/宽度脚本：epoch级 MAPE 计算使用反归一化后的 train/val 全量预测
        tr_hw_den = dl.denormalize_hw(tr["pred_hw_scaled"])
        tr_hw_t_den = dl.denormalize_hw(tr["tgt_hw_scaled"])
        va_hw_den = dl.denormalize_hw(va["pred_hw_scaled"])
        va_hw_t_den = dl.denormalize_hw(va["tgt_hw_scaled"])

        train_mape_height = compute_mape(tr_hw_t_den[:, 0], tr_hw_den[:, 0])
        train_mape_width = compute_mape(tr_hw_t_den[:, 1], tr_hw_den[:, 1])
        val_mape_height = compute_mape(va_hw_t_den[:, 0], va_hw_den[:, 0])
        val_mape_width = compute_mape(va_hw_t_den[:, 1], va_hw_den[:, 1])

        # 对齐颗粒脚本：epoch级 MSE/MAE 计算用反归一化后的预测
        tr_p_den = dl.denormalize_p(tr["pred_p_scaled"])
        tr_p_t_den = dl.denormalize_p(tr["tgt_p_scaled"])
        va_p_den = dl.denormalize_p(va["pred_p_scaled"])
        va_p_t_den = dl.denormalize_p(va["tgt_p_scaled"])

        tr_p_metrics = compute_metrics_particle(tr_p_t_den, tr_p_den)
        va_p_metrics = compute_metrics_particle(va_p_t_den, va_p_den)

        # scheduler 解耦：各自看各自的验证损失（对齐两套单任务脚本）
        sched.step(va["loss_total"])
        current_lr_shared = float(opt.param_groups[0]["lr"])
        current_lr_hw = float(opt.param_groups[1]["lr"])
        current_lr_p  = float(opt.param_groups[2]["lr"])

        # save best (分别保存 total / HW / P 三个最佳)
        if va["loss_total"] < best_val:
            best_val = va["loss_total"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "which": "total",
                "config": vars(args),
            }, best_path)

        if va["loss_hw"] < best_hw:
            best_hw = va["loss_hw"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val": best_hw,
                "which": "hw",
                "config": vars(args),
            }, best_hw_path)

        if va["loss_p"] < best_p:
            best_p = va["loss_p"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val": best_p,
                "which": "p",
                "config": vars(args),
            }, best_p_path)

        history_hw["train_loss"].append(tr["loss_hw"])
        history_hw["val_loss"].append(va["loss_hw"])
        history_hw["train_loss_height"].append(tr["loss_hw_h"])
        history_hw["train_loss_width"].append(tr["loss_hw_w"])
        history_hw["val_loss_height"].append(va["loss_hw_h"])
        history_hw["val_loss_width"].append(va["loss_hw_w"])
        history_hw["train_mape_height"].append(train_mape_height)
        history_hw["train_mape_width"].append(train_mape_width)
        history_hw["val_mape_height"].append(val_mape_height)
        history_hw["val_mape_width"].append(val_mape_width)
        history_hw["learning_rate"].append(current_lr_hw)
        history_hw["epoch"].append(epoch)

        history_p["train_loss"].append(tr["loss_p"])
        history_p["val_loss"].append(va["loss_p"])
        history_p["train_mse"].append(tr_p_metrics["mse"])
        history_p["val_mse"].append(va_p_metrics["mse"])
        history_p["train_mae"].append(tr_p_metrics["mae"])
        history_p["val_mae"].append(va_p_metrics["mae"])
        history_p["learning_rate"].append(current_lr_p)
        history_p["epoch"].append(epoch)

        # 直接记录训练/验证返回的多任务总 loss，包含颗粒分支及实际任务权重。
        history_total["epoch"].append(epoch)
        history_total["train_loss"].append(tr["loss_total"])
        history_total["val_loss"].append(va["loss_total"])

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            elapsed = (time.time() - start_time) / 60.0
            print(
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"HW loss={tr['loss_hw']:.6f}->{va['loss_hw']:.6f} | "
                f"P loss={tr['loss_p']:.6f}->{va['loss_p']:.6f} | "
                f"Total={tr['loss_total']:.6f}->{va['loss_total']:.6f} | "
                f"HW MAPE(H/W)={val_mape_height:.2f}%/{val_mape_width:.2f}% | "
                f"LR(shared/hw/p)={current_lr_shared:.2e}/{current_lr_hw:.2e}/{current_lr_p:.2e} | {elapsed:.1f} min"
            )

    # load best models & final test
    # 先用 best_hw 评估/出图 高度宽度；再用 best_p 评估/出图 颗粒占比

    # ---- HW (best_hw) ----
    ckpt_hw = torch.load(best_hw_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_hw["model_state_dict"])
    # 直接复用训练/验证的 FP32 loss 和按样本数加权汇总，loss 在缩放空间计算。
    with torch.no_grad():
        test_hw = run_epoch(
            model, test_loader, device, optimizer=None, scaler=None,
            hw_loss_weight=hw_loss_weight, p_loss_params=p_loss_params,
            w_hw=args.w_hw, w_p=args.w_p, amp=amp,
        )
    pred_hw_s = test_hw["pred_hw_scaled"]
    t_hw_s = test_hw["tgt_hw_scaled"]

    pred_hw = dl.denormalize_hw(pred_hw_s)
    t_hw = dl.denormalize_hw(t_hw_s)

    # 对齐高度/宽度测试指标（R²与MAPE）
    height_r2 = float(r2_score(t_hw[:, 0], pred_hw[:, 0]))
    width_r2 = float(r2_score(t_hw[:, 1], pred_hw[:, 1]))
    test_mape_height = compute_mape(t_hw[:, 0], pred_hw[:, 0])
    test_mape_width = compute_mape(t_hw[:, 1], pred_hw[:, 1])

    # ---- Particle (best_p) ----
    ckpt_p = torch.load(best_p_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_p["model_state_dict"])
    with torch.no_grad():
        test_p = run_epoch(
            model, test_loader, device, optimizer=None, scaler=None,
            hw_loss_weight=hw_loss_weight, p_loss_params=p_loss_params,
            w_hw=args.w_hw, w_p=args.w_p, amp=amp,
        )
    pred_p_s2 = test_p["pred_p_scaled"]
    t_p_s2 = test_p["tgt_p_scaled"]

    pred_p = dl.denormalize_p(pred_p_s2).reshape(-1)
    t_p = dl.denormalize_p(t_p_s2).reshape(-1)

    test_targets_denorm_all = t_p
    test_predictions_denorm_all = pred_p

    # 逐组预测（对齐“按组分割”+ 合并折线）
    all_group_targets = []
    all_group_preds = []
    try:
        g_imgs, g_hw_s, g_p_s = dl.get_group_test_data()
        for gi in range(len(g_imgs)):
            ds = torch.utils.data.TensorDataset(
                torch.tensor(g_imgs[gi], dtype=torch.float32),
                torch.tensor(g_hw_s[gi], dtype=torch.float32),
                torch.tensor(g_p_s[gi], dtype=torch.float32),
            )
            gl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
            phw_s_g, pp_s_g, thw_s_g, tp_s_g = predict(model, gl, device, amp=amp)

            pp_g = dl.denormalize_p(pp_s_g).reshape(-1)
            tp_g = dl.denormalize_p(tp_s_g).reshape(-1)

            all_group_targets.append(tp_g)
            all_group_preds.append(pp_g)
    except Exception as e:
        print(f"⚠️  分组测试数据获取/预测失败（不影响出图主流程）：{e}")
    # 计算测试集的详细指标
    eps = 1e-8

    # loss 与训练一致：缩放空间的 MSE；评价指标保留反归一化后的原始量纲。
    test_loss_height = test_hw["loss_hw_h"]
    test_loss_width = test_hw["loss_hw_w"]
    test_loss_hw = test_hw["loss_hw"]
    test_mae_height = float(np.mean(np.abs(pred_hw[:, 0] - t_hw[:, 0])))
    test_mae_width = float(np.mean(np.abs(pred_hw[:, 1] - t_hw[:, 1])))
    test_mse_height = float(np.mean((pred_hw[:, 0] - t_hw[:, 0]) ** 2))
    test_mse_width = float(np.mean((pred_hw[:, 1] - t_hw[:, 1]) ** 2))

    # 高度/宽度的最大相对误差
    height_rel_error = (pred_hw[:, 0] - t_hw[:, 0]) / (t_hw[:, 0] + eps)
    width_rel_error = (pred_hw[:, 1] - t_hw[:, 1]) / (t_hw[:, 1] + eps)
    max_relative_error_height = float(np.max(np.abs(height_rel_error)))
    max_relative_error_width = float(np.max(np.abs(width_rel_error)))

    # 颗粒占比的详细指标
    test_loss_particle = test_p["loss_p"]  # 与训练共用 SmoothL1 + MSE + MAE 及其配置权重
    test_mae_particle = float(np.mean(np.abs(pred_p - t_p)))
    test_mse_particle = float(np.mean((pred_p - t_p) ** 2))

    # 颗粒占比的最大相对误差
    particle_rel_error = (pred_p - t_p) / (t_p + eps)
    max_relative_error_particle = float(np.max(np.abs(particle_rel_error)))

    # 保留各任务使用各自最佳 checkpoint 的策略；这是两个 checkpoint 的组合结果。
    test_loss_total = combine_task_losses(test_loss_hw, test_loss_particle, args.w_hw, args.w_p)

    # 单独评估 best_total，提供可与训练/验证总 loss 对照的单一 checkpoint 结果。
    ckpt_total = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_total["model_state_dict"])
    with torch.no_grad():
        test_total = run_epoch(
            model, test_loader, device, optimizer=None, scaler=None,
            hw_loss_weight=hw_loss_weight, p_loss_params=p_loss_params,
            w_hw=args.w_hw, w_p=args.w_p, amp=amp,
        )
    best_total_test_losses = {
        key: test_total[key]
        for key in ("loss_total", "loss_hw", "loss_hw_h", "loss_hw_w", "loss_p")
    }
    print(
        f"\n[Test: per-task best] Height loss={test_loss_height:.6f} | "
        f"Width loss={test_loss_width:.6f} | HW loss={test_loss_hw:.6f} | "
        f"P loss={test_loss_particle:.6f} | Combined total={test_loss_total:.6f}"
    )
    print(f"[Test: best_total epoch={ckpt_total.get('epoch', -1)}] Total={test_total['loss_total']:.6f}")

    # 保存测试结果json（方便你后处理）
    results = {
        "best_epoch_hw": int(ckpt_hw.get("epoch", -1)),
        "best_val_hw": float(ckpt_hw.get("best_val", best_hw)),
        "best_epoch_total": int(ckpt_total.get("epoch", -1)),
        "best_val_total": float(ckpt_total.get("best_val", best_val)),
        "best_epoch_p": int(torch.load(best_p_path, map_location=device, weights_only=False).get("epoch", -1)),
        "best_val_p": float(torch.load(best_p_path, map_location=device, weights_only=False).get("best_val", best_p)),
        "height_r2": height_r2,
        "width_r2": width_r2,
        "height_mape": test_mape_height,
        "width_mape": test_mape_width,
        "particle_r2": float(r2_score(test_targets_denorm_all, test_predictions_denorm_all)) if len(test_targets_denorm_all) > 2 else -1.0,
        "particle_mape": compute_mape(test_targets_denorm_all, test_predictions_denorm_all),
        "test_metrics": {
            "height": {
                "loss": test_loss_height,
                "mae": test_mae_height,
                "mse": test_mse_height,
                "max_relative_error": max_relative_error_height,
            },
            "width": {
                "loss": test_loss_width,
                "mae": test_mae_width,
                "mse": test_mse_width,
                "max_relative_error": max_relative_error_width,
            },
            "particle": {
                "loss": test_loss_particle,
                "mae": test_mae_particle,
                "mse": test_mse_particle,
                "max_relative_error": max_relative_error_particle,
            },
            "loss_hw": test_loss_hw,
            "total_loss": test_loss_total,
        },
        "test_loss_protocol": {
            "space": "scaled_targets",
            "reduction": "sample_weighted_mean",
            "height_width_checkpoint": os.path.basename(best_hw_path),
            "particle_checkpoint": os.path.basename(best_p_path),
            "total_loss_checkpoint": "combined_per_task_best",
            "height_width_formula": "hw_loss_weight * MSE_height + (1 - hw_loss_weight) * MSE_width",
            "particle_formula": "huber * SmoothL1(beta=huber_beta) + mse * MSE + mae * MAE",
            "total_formula": TOTAL_LOSS_FORMULA,
            "total_loss_factor": TOTAL_LOSS_FACTOR,
            "hw_loss_weight": hw_loss_weight,
            "p_loss_params": p_loss_params,
            "w_hw": float(args.w_hw),
            "w_p": float(args.w_p),
            "metric_space": "original_units (mse/mae/r2/mape/relative_error)",
        },
        "best_total_test_losses": best_total_test_losses,
        "outdir": args.outdir,
    }
    with open(outpath(args.outdir, "multitask_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存训练历史json
    training_history = {
        "total": history_total,
        "height_width": history_hw,
        "particle": history_p,
        "best_epochs": {
            "total": int(torch.load(best_path, map_location=device, weights_only=False).get("epoch", -1)),
            "height_width": int(ckpt_hw.get("epoch", -1)),
            "particle": int(torch.load(best_p_path, map_location=device, weights_only=False).get("epoch", -1)),
        },
        "best_values": {
            "total": float(torch.load(best_path, map_location=device, weights_only=False).get("best_val", best_val)),
            "height_width": float(ckpt_hw.get("best_val", best_hw)),
            "particle": float(torch.load(best_p_path, map_location=device, weights_only=False).get("best_val", best_p)),
        },
        "test_metrics": {
            "losses_per_task_best": {
                "loss_hw_h": test_loss_height,
                "loss_hw_w": test_loss_width,
                "loss_hw": test_loss_hw,
                "loss_p": test_loss_particle,
                "loss_total": test_loss_total,
            },
            "losses_best_total": best_total_test_losses,
            "height_r2": height_r2,
            "width_r2": width_r2,
            "height_mape": test_mape_height,
            "width_mape": test_mape_width,
            "particle_r2": float(r2_score(test_targets_denorm_all, test_predictions_denorm_all)) if len(test_targets_denorm_all) > 2 else -1.0,
            "particle_mape": compute_mape(test_targets_denorm_all, test_predictions_denorm_all),
        },
    }
    with open(outpath(args.outdir, "multitask_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)

    # --- plots (对齐原脚本全部出图) ---
    # 关键：split_mode="global" 时，test_loader 的顺序是随机打乱过的（train_test_split 默认 shuffle）。
    # 为了让“台阶顺序严格按 group_id（真实工况顺序）”，我们只在 **绘图** 时对测试集按 (group_id, seq_id) 重排。
    t_hw_plot, pred_hw_plot = t_hw, pred_hw
    sorted_gids_plot = None
    if getattr(args, "split_mode", "").lower() in {"global", "overall", "all"}:
        if hasattr(dl, "test_gids") and hasattr(dl, "test_sids") and dl.test_gids is not None and dl.test_sids is not None:
            if len(dl.test_gids) == len(t_hw_plot) and len(dl.test_sids) == len(t_hw_plot):
                order = np.lexsort((dl.test_sids, dl.test_gids))  # 先按 gid，再按 sid
                t_hw_plot = t_hw_plot[order]
                pred_hw_plot = pred_hw_plot[order]
                sorted_gids_plot = dl.test_gids[order]
            else:
                print("⚠️  dl.test_gids/test_sids 长度与测试集不一致，已跳过按 group_id 重排绘图。")

    # 打印一下测试集里最大的相对误差点（便于定位你提到的尖峰 ≈0.44）
    try:
        eps = 1e-8
        h_rel = (pred_hw[:, 0] - t_hw[:, 0]) / (t_hw[:, 0] + eps)
        w_rel = (pred_hw[:, 1] - t_hw[:, 1]) / (t_hw[:, 1] + eps)
        ih = int(np.argmax(np.abs(h_rel)))
        iw = int(np.argmax(np.abs(w_rel)))
        gid_h = int(dl.test_gids[ih]) if hasattr(dl, "test_gids") and dl.test_gids is not None else -1
        sid_h = int(dl.test_sids[ih]) if hasattr(dl, "test_sids") and dl.test_sids is not None else -1
        gid_w = int(dl.test_gids[iw]) if hasattr(dl, "test_gids") and dl.test_gids is not None else -1
        sid_w = int(dl.test_sids[iw]) if hasattr(dl, "test_sids") and dl.test_sids is not None else -1
        print(
            f"\n🔎 最大|相对误差|样本：\n"
            f"  Height: |rel|={abs(h_rel[ih]):.4f} (rel={h_rel[ih]:.4f}) idx={ih} gid={gid_h} sid={sid_h} true={t_hw[ih,0]:.6f} pred={pred_hw[ih,0]:.6f}\n"
            f"  Width : |rel|={abs(w_rel[iw]):.4f} (rel={w_rel[iw]:.4f}) idx={iw} gid={gid_w} sid={sid_w} true={t_hw[iw,1]:.6f} pred={pred_hw[iw,1]:.6f}"
        )
    except Exception as _:
        pass

    plot_hw_all(history_hw, t_hw_plot, pred_hw_plot, height_r2, width_r2, test_mape_height, test_mape_width, args.outdir, sorted_gids=sorted_gids_plot)
    plot_total_loss(history_total, args.outdir)
    plot_particle_all(history_p, test_targets_denorm_all, test_predictions_denorm_all, all_group_targets, all_group_preds, args.outdir)


    # -------------------- 训练结束：导出“全量样本”预测 CSV（只生成一个文件） --------------------
    try:
        # 选择用于导出的 checkpoint
        ckpt_path = best_path
        if getattr(args, "pred_ckpt", "best") == "best_hw":
            ckpt_path = best_hw_path
        elif getattr(args, "pred_ckpt", "best") == "best_p":
            ckpt_path = best_p_path

        ckpt_any = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_any["model_state_dict"])

        # 构建全量数据 loader（不shuffle）
        all_loader, _, _ = dl.build_all_loader(batch_size=args.batch, num_workers=0, pin_memory=True)

        pred_hw_s_all, pred_p_s_all, _, _ = predict(model, all_loader, device, amp=amp)

        # 反归一化到原始量纲
        pred_hw_all = dl.denormalize_hw(pred_hw_s_all)
        pred_p_all = dl.denormalize_p(pred_p_s_all).reshape(-1)

        out_csv = outpath(args.outdir, getattr(args, "pred_csv", "all_predictions22.csv"))
        df_pred = pd.DataFrame({
            "height": pred_hw_all[:, 0],
            "width": pred_hw_all[:, 1],
            "particle": pred_p_all,
        })
        df_pred.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 已导出全量预测CSV: {os.path.abspath(out_csv)} (rows={len(df_pred)})")
    except Exception as e:
        print(f"⚠️ 导出全量预测CSV失败：{e}")

    print(f"\n✅ 出图完成。输出目录: {os.path.abspath(args.outdir)}")
    print(f"✅ 最佳模型: {os.path.abspath(best_path)}")
    print("✅ 已对齐原两套脚本的全部出图文件名与图形内容。")


if __name__ == "__main__":
    main()
