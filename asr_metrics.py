"""
asr_metrics.py - Metricas de reconhecimento de fala (ASR) para o projeto.

Inclui:
- Normalizacao de texto
- WER e CER por distancia de edicao (sem dependencia externa)
- Montagem de referencia a partir de transcricoes CHiME6 por intervalo de tempo
- Avaliacao de JSON de saida do baseline de transcricao
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Iterable, Optional


def _parse_time_to_seconds(value: Any) -> float:
    """Aceita float/int ou string no formato HH:MM:SS.sss."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" not in text:
        return float(text)
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparacao de ASR:
    - lower
    - remove tags entre []
    - remove pontuacao
    - compacta espacos
    """
    text = text.lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein_distance(seq_a: list[str], seq_b: list[str]) -> int:
    """Distancia de Levenshtein para listas de tokens."""
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, tok_a in enumerate(seq_a, start=1):
        cur = [i]
        for j, tok_b in enumerate(seq_b, start=1):
            cost = 0 if tok_a == tok_b else 1
            cur.append(min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            ))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> dict[str, float]:
    """Calcula WER retornando taxa e contagens basicas."""
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()

    edits = _levenshtein_distance(ref_tokens, hyp_tokens)
    denom = max(1, len(ref_tokens))
    wer = edits / denom

    return {
        "wer": wer,
        "edits": float(edits),
        "ref_words": float(len(ref_tokens)),
        "hyp_words": float(len(hyp_tokens)),
    }


def char_error_rate(reference: str, hypothesis: str) -> dict[str, float]:
    """Calcula CER retornando taxa e contagens basicas."""
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))

    edits = _levenshtein_distance(ref_chars, hyp_chars)
    denom = max(1, len(ref_chars))
    cer = edits / denom

    return {
        "cer": cer,
        "edits": float(edits),
        "ref_chars": float(len(ref_chars)),
        "hyp_chars": float(len(hyp_chars)),
    }


def _collect_reference_words(
    transcript_entries: Iterable[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[tuple[float, str]]:
    """
    Coleta palavras em um intervalo [start_sec, end_sec].
    Mantem segmentos que interceptam o intervalo.
    """
    selected: list[tuple[float, str]] = []
    for row in transcript_entries:
        seg_start = _parse_time_to_seconds(row.get("start_time", 0.0))
        seg_end = _parse_time_to_seconds(row.get("end_time", 0.0))

        if seg_end < start_sec or seg_start > end_sec:
            continue

        words = str(row.get("words", "")).strip()
        if not words:
            continue
        selected.append((seg_start, words))

    selected.sort(key=lambda x: x[0])
    return selected


def build_reference_text(
    transcript_json_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> str:
    """Monta texto de referencia concatenando segmentos ordenados no tempo."""
    with open(transcript_json_path, encoding="utf-8") as f:
        entries = json.load(f)

    pieces = _collect_reference_words(entries, start_sec=start_sec, end_sec=end_sec)
    return " ".join(text for _, text in pieces).strip()


def evaluate_asr_output(
    asr_json_path: str | Path,
    transcript_json_path: str | Path,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> dict[str, float]:
    """
    Avalia saida ASR salva em JSON contra transcricao de referencia.

    Se start/end nao forem fornecidos, usa os limites salvos no JSON ASR.
    """
    with open(asr_json_path, encoding="utf-8") as f:
        asr_data = json.load(f)

    asr_start = _parse_time_to_seconds(asr_data.get("start_sec", 0.0))
    asr_end = _parse_time_to_seconds(asr_data.get("end_sec", 0.0))

    start = asr_start if start_sec is None else float(start_sec)
    end = asr_end if end_sec is None else float(end_sec)

    ref = build_reference_text(transcript_json_path, start_sec=start, end_sec=end)
    hyp = str(asr_data.get("text", ""))

    wer_stats = word_error_rate(ref, hyp)
    cer_stats = char_error_rate(ref, hyp)

    return {
        "start_sec": float(start),
        "end_sec": float(end),
        "duration_sec": float(max(0.0, end - start)),
        **wer_stats,
        **cer_stats,
    }
