"""
Standard sequence data loader for the parallel CNN-LSTM multitask model.

This loader keeps the original temporal window as a sequence tensor:
    X:    (N, seq_length, num_features) = (N, 30, 8)
    y_hw: (N, 2)  -> height, width
    y_p:  (N, 1)  -> particle proportion

It intentionally does not convert sequences to 4-channel statistical images.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore")


class MultiTaskSequenceDataLoader:
    def __init__(
        self,
        file_path="换丝后所有成形数据.xlsx",
        seq_length=30,
        stride=30,
        image_size=None,
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        group_size=1800,
    ):
        self.file_path = file_path
        self.seq_length = int(seq_length)
        self.stride = int(stride)
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        self.image_size = image_size
        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.random_state = int(random_state)
        self.group_size = int(group_size)

        self.feature_scaler = RobustScaler()
        self.target_scaler_hw = RobustScaler()
        self.target_scaler_p = RobustScaler()
        self.feature_scaler_fitted = False
        self.target_scaler_hw_fitted = False
        self.target_scaler_p_fitted = False

        self.data_stats = {}

        self.train_gids = None
        self.val_gids = None
        self.test_gids = None
        self.train_sids = None
        self.val_sids = None
        self.test_sids = None

        self.all_test_sequences = []
        self.all_test_targets_hw = []
        self.all_test_targets_p = []

    def load_and_clean_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        df = pd.read_excel(self.file_path, header=None)

        feature_indices = list(range(1, 9))
        hw_target_indices = [9, 10]
        p_target_indices = [13]

        X_raw = df.iloc[1:, feature_indices].values
        y_hw_raw = df.iloc[1:, hw_target_indices].values
        y_p_raw = df.iloc[1:, p_target_indices].values

        self.data_stats["feature_columns"] = df.iloc[0, feature_indices].tolist()
        self.data_stats["target_columns_hw"] = df.iloc[0, hw_target_indices].tolist()
        self.data_stats["target_columns_p"] = df.iloc[0, p_target_indices].tolist()
        self.data_stats["original_data_shape"] = df.shape

        def _to_float_fill_mean(arr2d):
            arr2d = np.asarray(arr2d)
            out = np.zeros_like(arr2d, dtype=np.float32)
            for j in range(arr2d.shape[1]):
                col = pd.to_numeric(arr2d[:, j], errors="coerce").astype(np.float32)
                if np.isnan(col).all():
                    col = np.zeros_like(col, dtype=np.float32)
                elif np.isnan(col).any():
                    valid = col[~np.isnan(col)]
                    fill = float(np.mean(valid)) if len(valid) > 0 else 0.0
                    col[np.isnan(col)] = fill
                out[:, j] = col
            return out

        X = _to_float_fill_mean(X_raw)
        y_hw = _to_float_fill_mean(y_hw_raw)
        y_p = _to_float_fill_mean(y_p_raw)

        self.data_stats["processed_data_shape"] = (X.shape, y_hw.shape, y_p.shape)
        return X, y_hw, y_p, df

    def preprocess_features(self, X_raw, global_fit=False):
        X_raw = np.asarray(X_raw, dtype=np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        if global_fit or not self.feature_scaler_fitted:
            self.feature_scaler.fit(X_raw)
            self.feature_scaler_fitted = True

        X_scaled = self.feature_scaler.transform(X_raw).astype(np.float32)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        outlier_mask = np.ones(X_scaled.shape[0], dtype=bool)
        return X_scaled, outlier_mask

    def preprocess_targets_hw(self, y_hw_raw, outlier_mask, global_fit=False):
        y_hw_raw = np.asarray(y_hw_raw, dtype=np.float32)
        if len(outlier_mask) != len(y_hw_raw):
            outlier_mask = np.ones(len(y_hw_raw), dtype=bool)

        y_hw = np.nan_to_num(y_hw_raw[outlier_mask], nan=0.0, posinf=0.0, neginf=0.0)
        if global_fit or not self.target_scaler_hw_fitted:
            self.target_scaler_hw.fit(y_hw)
            self.target_scaler_hw_fitted = True

        y_hw_scaled = self.target_scaler_hw.transform(y_hw).astype(np.float32)
        y_hw_scaled = np.nan_to_num(y_hw_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return y_hw_scaled, y_hw

    def preprocess_targets_p(self, y_p_raw, outlier_mask, global_fit=False):
        y_p_raw = np.asarray(y_p_raw, dtype=np.float32)
        if y_p_raw.ndim == 1:
            y_p_raw = y_p_raw.reshape(-1, 1)
        if len(outlier_mask) != len(y_p_raw):
            outlier_mask = np.ones(len(y_p_raw), dtype=bool)

        y_p = np.nan_to_num(y_p_raw[outlier_mask], nan=0.0, posinf=0.0, neginf=0.0)
        if global_fit or not self.target_scaler_p_fitted:
            self.target_scaler_p.fit(y_p)
            self.target_scaler_p_fitted = True

        y_p_scaled = self.target_scaler_p.transform(y_p).astype(np.float32)
        y_p_scaled = np.nan_to_num(y_p_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return y_p_scaled, y_p

    def create_sequences(self, X, y_hw, y_p, include_partial=False):
        return self._make_sequence_samples(
            X,
            y_hw,
            y_p,
            seq_length=self.seq_length,
            stride=self.stride,
            include_partial=include_partial,
        )

    @staticmethod
    def _make_sequence_samples(group_X, group_y_hw, group_y_p, seq_length, stride=1, include_partial=False):
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

    @staticmethod
    def _loader_from_arrays(X, y_hw, y_p, batch_size, shuffle, num_workers=0, pin_memory=True):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_hw, dtype=torch.float32),
            torch.tensor(y_p, dtype=torch.float32),
        )
        return DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
        )

    def _build_all_full_window_sequences(self, X_scaled_all, y_hw_scaled_all, y_p_scaled_all):
        total_rows = len(X_scaled_all)
        num_groups = int(np.ceil(total_rows / self.group_size))

        all_X, all_y_hw, all_y_p, all_gid, all_sid = [], [], [], [], []
        per_group = []

        for g in range(num_groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, total_rows)
            Xg, yhg, ypg, sid = self.create_sequences(
                X_scaled_all[start:end],
                y_hw_scaled_all[start:end],
                y_p_scaled_all[start:end],
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
            raise ValueError("Not enough data to build sequence dataset.")

        return (
            np.concatenate(all_X, axis=0),
            np.concatenate(all_y_hw, axis=0),
            np.concatenate(all_y_p, axis=0),
            np.concatenate(all_gid, axis=0),
            np.concatenate(all_sid, axis=0),
            per_group,
        )

    def create_data_loaders(self, batch_size=32, num_workers=0, pin_memory=True, split_mode="chronological"):
        X_raw, y_hw_raw, y_p_raw, _ = self.load_and_clean_data()

        split_mode = "chronological" if split_mode is None else str(split_mode).lower().strip()
        if split_mode in {"chronological", "time", "temporal"}:
            return self._create_chronological_data_loaders(
                X_raw,
                y_hw_raw,
                y_p_raw,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        X_scaled_all, _ = self.preprocess_features(X_raw, global_fit=True)
        y_hw_scaled_all, _ = self.preprocess_targets_hw(
            y_hw_raw,
            np.ones(len(y_hw_raw), dtype=bool),
            global_fit=True,
        )
        y_p_scaled_all, _ = self.preprocess_targets_p(
            y_p_raw,
            np.ones(len(y_p_raw), dtype=bool),
            global_fit=True,
        )

        X_all, y_hw_all, y_p_all, gid_all, sid_all, per_group = self._build_all_full_window_sequences(
            X_scaled_all,
            y_hw_scaled_all,
            y_p_scaled_all,
        )

        if split_mode in {"global", "overall", "all"}:
            X_train, X_test, yhw_train, yhw_test, yp_train, yp_test, gid_train, gid_test, sid_train, sid_test = train_test_split(
                X_all,
                y_hw_all,
                y_p_all,
                gid_all,
                sid_all,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
            )
            val_ratio = self.val_size / max(1e-8, 1.0 - self.test_size)
            X_train, X_val, yhw_train, yhw_val, yp_train, yp_val, gid_train, gid_val, sid_train, sid_val = train_test_split(
                X_train,
                yhw_train,
                yp_train,
                gid_train,
                sid_train,
                test_size=val_ratio,
                random_state=self.random_state,
                shuffle=True,
            )
        elif split_mode in {"by_group", "group"}:
            train_parts, val_parts, test_parts = [], [], []
            for Xg, yhg, ypg, gid, sid in per_group:
                if len(Xg) < 3:
                    continue
                X_tr, X_te, yhw_tr, yhw_te, yp_tr, yp_te, gid_tr, gid_te, sid_tr, sid_te = train_test_split(
                    Xg,
                    yhg,
                    ypg,
                    gid,
                    sid,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=True,
                )
                val_ratio = self.val_size / max(1e-8, 1.0 - self.test_size)
                X_tr, X_va, yhw_tr, yhw_va, yp_tr, yp_va, gid_tr, gid_va, sid_tr, sid_va = train_test_split(
                    X_tr,
                    yhw_tr,
                    yp_tr,
                    gid_tr,
                    sid_tr,
                    test_size=val_ratio,
                    random_state=self.random_state,
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

        self.train_gids = gid_train.astype(np.int32)
        self.val_gids = gid_val.astype(np.int32)
        self.test_gids = gid_test.astype(np.int32)
        self.train_sids = sid_train.astype(np.int32)
        self.val_sids = sid_val.astype(np.int32)
        self.test_sids = sid_test.astype(np.int32)

        self.all_test_sequences = []
        self.all_test_targets_hw = []
        self.all_test_targets_p = []
        for g in sorted(set(gid_test.tolist())):
            idx = gid_test == g
            order = np.argsort(sid_test[idx], kind="stable")
            self.all_test_sequences.append(X_test[idx][order].astype(np.float32))
            self.all_test_targets_hw.append(yhw_test[idx][order].astype(np.float32))
            self.all_test_targets_p.append(yp_test[idx][order].astype(np.float32))

        self.data_stats.update(
            {
                "total_rows": int(len(X_raw)),
                "num_groups": int(np.ceil(len(X_raw) / self.group_size)),
                "group_size_rows": int(self.group_size),
                "total_sequences": int(len(X_all)),
                "train_size": int(len(X_train)),
                "val_size": int(len(X_val)),
                "test_size": int(len(X_test)),
                "seq_length": int(self.seq_length),
                "stride": int(self.stride),
                "num_features": int(X_all.shape[-1]),
                "input_shape": "(batch, seq_length, num_features)",
                "split_mode": split_mode,
            }
        )

        return (
            self._loader_from_arrays(X_train, yhw_train, yp_train, batch_size, True, num_workers, pin_memory),
            self._loader_from_arrays(X_val, yhw_val, yp_val, batch_size, False, num_workers, pin_memory),
            self._loader_from_arrays(X_test, yhw_test, yp_test, batch_size, False, num_workers, pin_memory),
        )

    def _create_chronological_data_loaders(
        self,
        X_raw,
        y_hw_raw,
        y_p_raw,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    ):
        """Build windows inside each group and assign them by their target time."""
        train_ratio = 1.0 - self.test_size - self.val_size
        if train_ratio <= 0.0:
            raise ValueError("test_size + val_size must be smaller than 1.0")

        total_rows = len(X_raw)
        num_groups = int(np.ceil(total_rows / self.group_size))
        group_slices = []
        train_row_indices = []

        for g in range(num_groups):
            group_start = g * self.group_size
            group_end = min((g + 1) * self.group_size, total_rows)
            group_len = group_end - group_start

            train_len = int(np.floor(group_len * train_ratio))
            test_len = int(np.floor(group_len * self.test_size))
            val_len = group_len - train_len - test_len
            train_end = group_start + train_len
            test_end = train_end + test_len

            group_slices.append((g, group_start, train_end, test_end, group_end))
            train_row_indices.extend(range(group_start, train_end))

            if train_len < self.seq_length or test_len == 0 or val_len == 0:
                warnings.warn(
                    f"Group {g}: chronological row partitions are "
                    f"train/test/val={train_len}/{test_len}/{val_len}; "
                    f"training requires at least seq_length={self.seq_length} rows, "
                    "and test/validation must each contain at least one target row."
                )

        if not train_row_indices:
            raise ValueError("No training rows are available after chronological splitting.")

        train_row_indices = np.asarray(train_row_indices, dtype=np.int64)

        # Fit every scaler on training rows only, then transform the complete data.
        X_clean = np.nan_to_num(np.asarray(X_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_hw_clean = np.nan_to_num(np.asarray(y_hw_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_p_clean = np.nan_to_num(np.asarray(y_p_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_scaler.fit(X_clean[train_row_indices])
        self.target_scaler_hw.fit(y_hw_clean[train_row_indices])
        self.target_scaler_p.fit(y_p_clean[train_row_indices])
        self.feature_scaler_fitted = True
        self.target_scaler_hw_fitted = True
        self.target_scaler_p_fitted = True

        X_scaled_all = self.feature_scaler.transform(X_clean).astype(np.float32)
        y_hw_scaled_all = self.target_scaler_hw.transform(y_hw_clean).astype(np.float32)
        y_p_scaled_all = self.target_scaler_p.transform(y_p_clean).astype(np.float32)

        train_parts, test_parts, val_parts = [], [], []

        def _append_by_mask(parts, Xg, yhg, ypg, gid, sid, mask):
            if not np.any(mask):
                return
            parts.append((Xg[mask], yhg[mask], ypg[mask], gid[mask], sid[mask]))

        for g, group_start, train_end, test_end, group_end in group_slices:
            # Slide only inside the current 300-row group. A sample is assigned
            # according to the time index of its target (the window's last row).
            Xg, yhg, ypg, sid = self.create_sequences(
                X_scaled_all[group_start:group_end],
                y_hw_scaled_all[group_start:group_end],
                y_p_scaled_all[group_start:group_end],
                include_partial=False,
            )
            if Xg is None or len(Xg) == 0:
                continue

            train_cut = train_end - group_start
            test_cut = test_end - group_start
            gid = np.full((len(Xg),), g, dtype=np.int32)
            _append_by_mask(train_parts, Xg, yhg, ypg, gid, sid, sid < train_cut)
            _append_by_mask(test_parts, Xg, yhg, ypg, gid, sid, (sid >= train_cut) & (sid < test_cut))
            _append_by_mask(val_parts, Xg, yhg, ypg, gid, sid, sid >= test_cut)

        if not train_parts or not test_parts or not val_parts:
            raise ValueError(
                "Chronological split produced an empty dataset; check group_size, "
                "seq_length, test_size, val_size, and the final partial group."
            )

        def _cat(parts, pos):
            return np.concatenate([p[pos] for p in parts], axis=0)

        X_train, yhw_train, yp_train, gid_train, sid_train = (_cat(train_parts, i) for i in range(5))
        X_test, yhw_test, yp_test, gid_test, sid_test = (_cat(test_parts, i) for i in range(5))
        X_val, yhw_val, yp_val, gid_val, sid_val = (_cat(val_parts, i) for i in range(5))

        self.train_gids = gid_train.astype(np.int32)
        self.test_gids = gid_test.astype(np.int32)
        self.val_gids = gid_val.astype(np.int32)
        self.train_sids = sid_train.astype(np.int32)
        self.test_sids = sid_test.astype(np.int32)
        self.val_sids = sid_val.astype(np.int32)

        self.all_test_sequences = []
        self.all_test_targets_hw = []
        self.all_test_targets_p = []
        for g in sorted(set(gid_test.tolist())):
            idx = gid_test == g
            order = np.argsort(sid_test[idx], kind="stable")
            self.all_test_sequences.append(X_test[idx][order].astype(np.float32))
            self.all_test_targets_hw.append(yhw_test[idx][order].astype(np.float32))
            self.all_test_targets_p.append(yp_test[idx][order].astype(np.float32))

        self.data_stats.update(
            {
                "total_rows": int(total_rows),
                "num_groups": int(num_groups),
                "group_size_rows": int(self.group_size),
                "total_sequences": int(len(X_train) + len(X_test) + len(X_val)),
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "val_size": int(len(X_val)),
                "seq_length": int(self.seq_length),
                "stride": int(self.stride),
                "num_features": int(X_train.shape[-1]),
                "input_shape": "(batch, seq_length, num_features)",
                "split_mode": "chronological",
                "row_split_order": "train_test_val",
                "row_split_ratio": [float(train_ratio), float(self.test_size), float(self.val_size)],
                "window_assignment": "target_end_time",
            }
        )

        return (
            self._loader_from_arrays(X_train, yhw_train, yp_train, batch_size, True, num_workers, pin_memory),
            self._loader_from_arrays(X_val, yhw_val, yp_val, batch_size, False, num_workers, pin_memory),
            self._loader_from_arrays(X_test, yhw_test, yp_test, batch_size, False, num_workers, pin_memory),
        )

    def build_all_loader(self, batch_size=32, num_workers=0, pin_memory=True):
        X_raw, y_hw_raw, y_p_raw, _ = self.load_and_clean_data()
        X_scaled_all, _ = self.preprocess_features(X_raw, global_fit=False)
        y_hw_scaled_all, _ = self.preprocess_targets_hw(
            y_hw_raw,
            np.ones(len(y_hw_raw), dtype=bool),
            global_fit=False,
        )
        y_p_scaled_all, _ = self.preprocess_targets_p(
            y_p_raw,
            np.ones(len(y_p_raw), dtype=bool),
            global_fit=False,
        )

        total_rows = len(X_raw)
        num_groups = int(np.ceil(total_rows / self.group_size))
        all_X, all_y_hw, all_y_p, all_gid, all_sid = [], [], [], [], []

        for g in range(num_groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, total_rows)
            Xg, yhg, ypg, sid = self.create_sequences(
                X_scaled_all[start:end],
                y_hw_scaled_all[start:end],
                y_p_scaled_all[start:end],
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
            raise ValueError("Unable to build full-row sequence dataset.")

        X_all = np.concatenate(all_X, axis=0)
        y_hw_all = np.concatenate(all_y_hw, axis=0)
        y_p_all = np.concatenate(all_y_p, axis=0)
        gid_all = np.concatenate(all_gid, axis=0)
        sid_all = np.concatenate(all_sid, axis=0)

        return (
            self._loader_from_arrays(X_all, y_hw_all, y_p_all, batch_size, False, num_workers, pin_memory),
            gid_all,
            sid_all,
        )

    def get_group_test_data(self):
        return self.all_test_sequences, self.all_test_targets_hw, self.all_test_targets_p

    def save_preprocessing_params(self, filepath="multitask_sequence_preprocessing_params.pt"):
        params = {
            "feature_scaler": self.feature_scaler,
            "target_scaler_hw": self.target_scaler_hw,
            "target_scaler_p": self.target_scaler_p,
            "data_stats": self.data_stats,
            "config": {
                "file_path": self.file_path,
                "seq_length": self.seq_length,
                "stride": self.stride,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "random_state": self.random_state,
                "group_size": self.group_size,
            },
        }
        torch.save(params, filepath)

    def load_preprocessing_params(self, filepath="multitask_sequence_preprocessing_params.pt"):
        params = torch.load(filepath, map_location="cpu", weights_only=False)
        self.feature_scaler = params["feature_scaler"]
        self.target_scaler_hw = params["target_scaler_hw"]
        self.target_scaler_p = params["target_scaler_p"]
        self.feature_scaler_fitted = True
        self.target_scaler_hw_fitted = True
        self.target_scaler_p_fitted = True

        cfg = params.get("config", {})
        self.file_path = cfg.get("file_path", self.file_path)
        self.seq_length = int(cfg.get("seq_length", self.seq_length))
        self.stride = int(cfg.get("stride", self.stride))
        self.test_size = float(cfg.get("test_size", self.test_size))
        self.val_size = float(cfg.get("val_size", self.val_size))
        self.random_state = int(cfg.get("random_state", self.random_state))
        self.group_size = int(cfg.get("group_size", self.group_size))
        self.data_stats = params.get("data_stats", {})

    def denormalize_hw(self, y_scaled):
        y_scaled = np.asarray(y_scaled)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler_hw.inverse_transform(y_scaled)

    def denormalize_p(self, y_scaled):
        y_scaled = np.asarray(y_scaled)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler_p.inverse_transform(y_scaled)


# Backward-compatible alias for training scripts that import MultiTaskBNDataLoader.
MultiTaskBNDataLoader = MultiTaskSequenceDataLoader
