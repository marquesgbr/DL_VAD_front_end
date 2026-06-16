"""
transcription.py — baseline de transcrição ASR para os áudios do CHiME6.

Foco:
- Executar transcrição sem depender de ffmpeg externo
- Processar áudios longos em janelas (chunks)
- Permitir transcrição por faixa de tempo (ex.: primeira 1h de S09)

Backend ASR:
- Hugging Face Transformers (Whisper)

Observação:
- O objetivo aqui é ter uma referência reproduzível e leve de integração.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import librosa
import torch
from transformers import pipeline


DEFAULT_MODEL_ID = "openai/whisper-small"
DEFAULT_SR = 16000


@dataclass
class TranscriptionResult:
    file_path: str
    model_id: str
    language: str
    start_sec: float
    end_sec: float
    text: str
    segments: list[dict]
    processing_time: float


def get_device() -> int:
    """Retorna índice CUDA (0) se disponível, caso contrário CPU (-1)."""
    return 0 if torch.cuda.is_available() else -1

def parse_seconds(seconds: float) -> str:
    """Converter segundos para formato hh:mm:ss, útil para ffmpeg."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60) + float(seconds % 1)
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"


def build_transcriber(
    model_id: str = DEFAULT_MODEL_ID,
    language: str = "en",
    task: str = "transcribe",
):
    """
    Cria pipeline de ASR (Whisper) para transcrição com timestamps.

    Parameters
    ----------
    model_id : modelo Whisper no Hugging Face Hub
    language : idioma alvo da transcrição (ex.: 'en')
    task     : 'transcribe' ou 'translate'
    """
    device = get_device()
    dtype = torch.float16 if device >= 0 else torch.float32

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        dtype=dtype,
        device=device,
    )

    # Modelos Whisper English-only (ex.: *.en) não aceitam language/task.
    # Para modelos multilíngues, passamos language/task explicitamente.
    model_name = str(model_id).lower()
    english_only_hint = model_name.endswith(".en")
    is_multilingual = bool(getattr(asr.model.config, "is_multilingual", True)) and not english_only_hint
    if is_multilingual:
        generate_kwargs = {"language": language, "task": task}
    else:
        generate_kwargs = {}

    return asr, generate_kwargs


def _load_audio_slice(
    wav_path: str | Path,
    sr: int,
    start_sec: float,
    duration_sec: float,
) -> np.ndarray:
    """Carrega apenas uma fatia do áudio para reduzir uso de RAM."""
    audio, _ = librosa.load(wav_path, sr=sr, mono=True, offset=start_sec, duration=duration_sec)
    return audio.astype(np.float32)


def transcribe_range(
    asr,
    generate_kwargs: dict,
    wav_path: str | Path,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    sr: int = DEFAULT_SR,
    chunk_sec: int = 30,
    batch_size: int = 8,
    chunk_batch_size: Optional[int] = None,
) -> TranscriptionResult:
    """
    Transcreve uma faixa de tempo do arquivo em blocos fixos.

    Parameters
    ----------
    start_sec : início da faixa (segundos)
    end_sec   : fim da faixa (segundos). Se None, vai até o fim do arquivo
    chunk_sec : tamanho de bloco de leitura/decodificação
    batch_size: batch interno do modelo na pipeline
    chunk_batch_size: quantidade de chunks de áudio enviados em uma única
                      chamada da pipeline. Maior valor tende a usar melhor a GPU.
    """
    wav_path = Path(wav_path)

    if end_sec is not None and end_sec <= start_sec:
        raise ValueError("end_sec deve ser maior que start_sec")

    # Descobre duração total sem carregar o áudio inteiro.
    total_duration = librosa.get_duration(path=str(wav_path))
    final_end = min(end_sec, total_duration) if end_sec is not None else total_duration

    all_segments: list[dict] = []
    all_text_parts: list[str] = []

    cursor = float(start_sec)
    if chunk_batch_size is None:
        chunk_batch_size = max(1, int(batch_size))

    # calcular tempo de processamento aproximado para feedback
    start_time = time()
    while cursor < final_end:
        batch_inputs: list[dict] = []
        batch_offsets: list[float] = []
        batch_durations: list[float] = []

        for _ in range(chunk_batch_size):
            if cursor >= final_end:
                break

            dur = min(float(chunk_sec), final_end - cursor)
            audio = _load_audio_slice(wav_path, sr=sr, start_sec=cursor, duration_sec=dur)
            if len(audio) == 0:
                break

            batch_inputs.append({"raw": audio, "sampling_rate": sr})
            batch_offsets.append(cursor)
            batch_durations.append(dur)
            cursor += dur

        if not batch_inputs:
            break

        results = asr(
            batch_inputs,
            return_timestamps=True,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
        )

        if isinstance(results, dict):
            results = [results]

        for result, offset, dur in zip(results, batch_offsets, batch_durations):
            # `chunks` inclui timestamps locais ao trecho, então deslocamos por `offset`.
            chunks = result.get("chunks", [])
            for ch in chunks:
                ts = ch.get("timestamp")
                if not ts:
                    continue
                seg_start = float(ts[0] + offset)
                seg_end = float(ts[1] + offset) if ts[1] is not None else float(offset + dur)
                seg_text = ch.get("text", "").strip()
                if seg_text:
                    all_segments.append(
                        {
                            "start_sec": seg_start,
                            "end_sec": seg_end,
                            "text": seg_text,
                        }
                    )

            txt = result.get("text", "").strip()
            if txt:
                all_text_parts.append(txt)

    end_time = time()
    elapsed = end_time - start_time
    full_text = " ".join(all_text_parts).strip()

    return TranscriptionResult(
        file_path=str(wav_path),
        model_id=getattr(asr.model.config, "_name_or_path", "unknown"),
        language=generate_kwargs.get("language", "unknown"),
        start_sec=float(start_sec),
        end_sec=float(final_end),
        text=full_text,
        segments=all_segments,
        processing_time=elapsed,
    )


def save_transcription_json(result: TranscriptionResult, output_path: str | Path) -> None:
    """Salva resultado da transcrição em JSON."""
    import json

    payload = {
        "file_path": result.file_path,
        "model_id": result.model_id,
        "language": result.language,
        "start_sec": parse_seconds(result.start_sec),
        "end_sec": parse_seconds(result.end_sec),
        "processing_time_sec": result.processing_time,
        "text": result.text,
        "segments": result.segments,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_session_files(
    audio_dir: str | Path,
    session_id: str,
    contains: Optional[str] = None,
) -> list[Path]:
    """Lista WAVs de uma sessão, opcionalmente filtrando por substring."""
    audio_dir = Path(audio_dir)
    files = sorted(audio_dir.glob(f"{session_id}_*.wav"))
    if contains:
        files = [p for p in files if contains in p.name]
    return files


def split_s09_ranges(duration: float = 3600.0) -> dict[str, tuple[float, Optional[float]]]:
    """
    Define ranges da sessão S09 conforme estratégia do projeto.

    - val: primeira 1h
    - test: resto da sessão
    """
    return {
        "val": (0.0, min(3600.0, duration)),
        "test": (min(3600.0, duration), duration if duration > 3600.0 else None),
    }
