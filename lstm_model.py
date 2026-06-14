"""
lstm_model.py — Arquitetura LSTM e utilitários de treino para classificação de records de áudio.

Responsabilidades:
  - SequenceDataset: Dataset PyTorch para records (X: 3D, y: 1D)
  - LSTMClassifier: modelo LSTM binário configurável
  - train_sequence_model: loop de treino com early stopping por val_loss
  - evaluate_model: avaliação completa (accuracy, precision, recall, f1, auroc)
  - infer_records: inferência em lote, retorna preds e scores por record
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Dataset para records sequenciais.

    Parameters
    ----------
    X : np.ndarray  shape (n_records, seq_len, n_features)
    y : np.ndarray  shape (n_records,), valores 0 ou 1
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    LSTM binário para classificação de records de áudio (fala vs. ruído).

    Architecture:
      LSTM (stacked, batch_first) → dropout → FC1 → ReLU → FC2 → logit

    Parameters
    ----------
    input_dim    : n_features por frame (saída do preprocess.py)
    hidden_dim   : dimensão do estado oculto do LSTM
    num_layers   : número de camadas LSTM empilhadas
    dropout_rate : dropout aplicado entre camadas e antes do head FC
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        if output_dim not in (1, 2):
            raise ValueError("output_dim must be 1 or 2 for binary classification.")
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : shape (batch, seq_len, input_dim)

        Returns
        -------
        logits : shape (batch, output_dim)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_dim)
        last = lstm_out[:, -1, :]           # último timestep
        out = self.dropout(last)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(SequenceDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def _encode_binary_labels(y: Any) -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in ("U", "S", "O"):
        mapped = []
        for value in y_arr:
            key = str(value).strip().lower()
            if key in ("yes", "true", "1"):
                mapped.append(1)
            elif key in ("no", "false", "0"):
                mapped.append(0)
            else:
                raise ValueError(f"Unsupported label value: {value}")
        return np.asarray(mapped, dtype=np.int64)
    return y_arr.astype(np.int64)


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Computa métricas binárias padronizadas para comparação de modelos.
    """
    y_true_arr = _encode_binary_labels(y_true)
    y_pred_arr = _encode_binary_labels(y_pred)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    if y_score is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true_arr, np.asarray(y_score)))
        except ValueError:
            metrics["auroc"] = float("nan")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    return metrics


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train_sequence_model(
    model: LSTMClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    min_delta: float = 1e-4,
    device: Optional[torch.device] = None,
    class_weights: Optional[torch.Tensor] = None,
    loss_name: Optional[str] = None,
) -> dict:
    """
    Treina o LSTMClassifier com early stopping por val_loss.

    Parameters
    ----------
    model         : instância de LSTMClassifier
    X_train/X_val : shape (n, seq_len, n_features)
    y_train/y_val : shape (n,)
    epochs        : número máximo de épocas
    batch_size    : tamanho do mini-batch
    lr            : learning rate do Adam
    patience      : épocas sem melhora na val_loss antes de parar
    min_delta     : redução mínima na val_loss para resetar patience
    class_weights : tensor de pesos por classe (shape (2,)) ou None

    Returns
    -------
    dict com chaves:

      model      : modelo com melhores pesos carregados
      history    : {train_loss: [...], val_loss: [...]}
      best_val_loss : menor val_loss atingida
      epochs_run : épocas efetivamente rodadas
    """
    device = device or _get_device()
    model.to(device)

    resolved_loss = loss_name
    if resolved_loss is None:
        resolved_loss = "bce_with_logits" if model.output_dim == 1 else "cross_entropy"

    if resolved_loss == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
    elif resolved_loss == "bce_with_logits":
        pos_weight = None
        if class_weights is not None:
            cw = class_weights.detach().cpu().numpy()
            if cw.shape[0] >= 2 and cw[0] > 0:
                pos_weight = torch.tensor(float(cw[1] / cw[0]), dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError("loss_name must be 'cross_entropy' or 'bce_with_logits'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False)

    history: dict = {"train_loss": [], "val_loss": []}
    best_val_loss = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- treino ---
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            if resolved_loss == "bce_with_logits":
                loss = loss_fn(logits.squeeze(-1), batch_y.float())
            else:
                loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_y)
        history["train_loss"].append(running_loss / len(X_train))

        # --- validação ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                if resolved_loss == "bce_with_logits":
                    batch_loss = loss_fn(logits.squeeze(-1), batch_y.float())
                else:
                    batch_loss = loss_fn(logits, batch_y)
                val_running_loss += batch_loss.item() * len(batch_y)
        val_loss = val_running_loss / len(X_val)
        history["val_loss"].append(val_loss)

        # --- early stopping por val_loss ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping na época {epoch + 1}  (best val_loss={best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model":         model,
        "history":       history,
        "best_val_loss": best_val_loss,
        "epochs_run":    epoch + 1,
    }


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LSTMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "val",
    batch_size: int = 256,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Avalia o modelo em um conjunto e retorna todas as métricas de classificação.

    Parameters
    ----------
    X          : shape (n_records, seq_len, n_features)
    y          : shape (n_records,), ground-truth labels
    split_name : nome do conjunto (ex.: 'train', 'val', 'evaluation', 'test')
    threshold  : limiar de probabilidade para classe 1

    Returns
    -------
    dict com chaves:
      split, n_records, accuracy, precision, recall, f1, auroc,
      confusion_matrix (tn, fp, fn, tp)
    """
    preds, scores = infer_records(model, X, batch_size=batch_size,
                                   threshold=threshold, device=device)
    y_true = np.asarray(y)
    metrics = compute_metrics(y_true, preds, scores)
    return {
        "split": split_name,
        "n_records": int(len(y_true)),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

def infer_records(
    model: LSTMClassifier,
    X: np.ndarray,
    batch_size: int = 256,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classifica records em lote.

    Parameters
    ----------
    X         : shape (n_records, seq_len, n_features)
    threshold : limiar sobre a probabilidade da classe 1

    Returns
    -------
    preds  : np.ndarray shape (n_records,), dtype int  (0 ou 1)
    scores : np.ndarray shape (n_records,), dtype float  (prob. classe 1)
    """
    device = device or _get_device()
    model.eval()
    model.to(device)

    # dataset = SequenceDataset(X, np.zeros(len(X), dtype=np.int64))
    # loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loader = _make_loader(X, np.zeros(len(X), dtype=np.int64), batch_size, shuffle=False)

    all_scores = []
    with torch.no_grad():
        for batch_X, _ in loader:
            logits = model(batch_X.to(device))
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                probs = torch.sigmoid(logits.squeeze(-1))
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    preds  = (scores >= threshold).astype(int)
    return preds, scores


# ---------------------------------------------------------------------------
# Cálculo de pesos de classe (para desbalanceamento)
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe.
    Útil quando há muito mais ruído (0) do que fala (1) ou vice-versa.

    Returns
    -------
    weights : torch.Tensor shape (2,)
    """
    counts = np.bincount(y, minlength=2).astype(np.float32)
    total  = counts.sum()
    weights = total / (2.0 * counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Serialização do modelo
# ---------------------------------------------------------------------------

def save_model(model: LSTMClassifier, path: str) -> None:
    """Salva apenas os pesos (state_dict) do modelo."""
    torch.save(model.state_dict(), path)


def load_model(model: LSTMClassifier, path: str, device: Optional[torch.device] = None) -> LSTMClassifier:
    """Carrega pesos salvos em um modelo com a mesma arquitetura."""
    device = device or _get_device()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
