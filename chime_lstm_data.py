"""
chime_lstm_data.py - Utilitarios para treino temporal de LSTM na base CHiME6.

Foco:
- Carregar CSV de features (log_mel + spatial + label)
- Garantir splits temporais sem embaralhar a ordem
- Gerar sequencias (records) para LSTM com metadados de tempo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SequencePack:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def load_feature_csv(path: str, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Carrega CSV e padroniza ordenacao temporal minima."""
    df = pd.read_csv(path, usecols=usecols)
    required = {"sample_id", "timestamp_ms", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(missing)}")

    df = df.sort_values(["sample_id", "timestamp_ms"]).reset_index(drop=True)
    return df


def detect_feature_columns(df: pd.DataFrame) -> list[str]:
    """Seleciona colunas de feature esperadas (log_mel_* e spatial_*)."""
    cols = [c for c in df.columns if c.startswith("log_mel_") or c.startswith("spatial_")]
    if not cols:
        raise ValueError("Nenhuma coluna de feature detectada (log_mel_/spatial_).")
    return cols


def temporal_split_by_group(
    df: pd.DataFrame,
    train_ratio: float,
    group_col: str = "sample_id",
    time_col: str = "timestamp_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal por grupo (sem shuffle): inicio -> train, fim -> holdout.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio deve estar em (0, 1).")

    train_parts = []
    hold_parts = []
    for _, g in df.groupby(group_col, sort=False):
        g = g.sort_values(time_col)
        cut = int(len(g) * train_ratio)
        cut = max(1, min(cut, len(g) - 1))
        train_parts.append(g.iloc[:cut])
        hold_parts.append(g.iloc[cut:])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    hold_df = pd.concat(hold_parts, axis=0).reset_index(drop=True)
    return train_df, hold_df


def contiguous_subsample_by_group(
    df: pd.DataFrame,
    frac: float,
    group_col: str = "sample_id",
    time_col: str = "timestamp_ms",
    from_start: bool = True,
) -> pd.DataFrame:
    """
    Subamostra temporal continua por grupo.

    Nao embaralha linhas; pega um bloco contiguo por sample_id.
    """
    if frac >= 1.0:
        return df.copy()
    if frac <= 0.0:
        raise ValueError("frac deve ser > 0.")

    parts = []
    for _, g in df.groupby(group_col, sort=False):
        g = g.sort_values(time_col)
        n = max(1, int(len(g) * frac))
        if from_start:
            parts.append(g.iloc[:n])
        else:
            parts.append(g.iloc[-n:])
    return pd.concat(parts, axis=0).reset_index(drop=True)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 150,
    stride: int = 150,
    group_col: str = "sample_id",
    time_col: str = "timestamp_ms",
    label_col: str = "label",
    label_mode: str = "majority",
) -> SequencePack:
    """
    Converte dataframe temporal em records para LSTM.

    label_mode:
    - majority: label da janela por maioria
    - last: label do ultimo frame da janela
    """
    if seq_len <= 1:
        raise ValueError("seq_len deve ser > 1")
    if stride <= 0:
        raise ValueError("stride deve ser > 0")

    X_list = []
    y_list = []
    meta_rows = []

    for sid, g in df.groupby(group_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        Xg = g[feature_cols].to_numpy(dtype=np.float32)
        yg = g[label_col].to_numpy(dtype=np.int64)
        tg = g[time_col].to_numpy(dtype=np.float64)

        if len(g) < seq_len:
            continue

        for start in range(0, len(g) - seq_len + 1, stride):
            end = start + seq_len
            Xw = Xg[start:end]
            yw = yg[start:end]

            if label_mode == "majority":
                y_label = int(np.mean(yw) >= 0.5)
            elif label_mode == "last":
                y_label = int(yw[-1])
            else:
                raise ValueError("label_mode invalido. Use 'majority' ou 'last'.")

            X_list.append(Xw)
            y_list.append(y_label)
            meta_rows.append(
                {
                    "sample_id": sid,
                    "start_timestamp_ms": float(tg[start]),
                    "end_timestamp_ms": float(tg[end - 1]),
                }
            )

    if not X_list:
        n_features = len(feature_cols)
        return SequencePack(
            X=np.empty((0, seq_len, n_features), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            meta=pd.DataFrame(columns=["sample_id", "start_timestamp_ms", "end_timestamp_ms"]),
        )

    return SequencePack(
        X=np.asarray(X_list, dtype=np.float32),
        y=np.asarray(y_list, dtype=np.int64),
        meta=pd.DataFrame(meta_rows),
    )


def split_eval_half_timeline(
    df_eval: pd.DataFrame,
    group_col: str = "sample_id",
    time_col: str = "timestamp_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide cada serie do eval ao meio no tempo:
    - primeira metade: evaluation
    - segunda metade: test
    """
    eval_parts = []
    test_parts = []

    for _, g in df_eval.groupby(group_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        cut = len(g) // 2
        cut = max(1, min(cut, len(g) - 1))
        eval_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])

    return (
        pd.concat(eval_parts, axis=0).reset_index(drop=True),
        pd.concat(test_parts, axis=0).reset_index(drop=True),
    )
