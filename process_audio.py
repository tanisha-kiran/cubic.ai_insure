"""
Layer 1 — Audio Ingestion (faster-whisper)

Transcribes an audio file locally (no cloud dependency) and normalizes the
output into the same StructuredContent schema used by process_backend.py,
so downstream Layer 2 (reasoning) can consume audio and documents uniformly.

Usage:
    python process_audio.py path/to/audio.mp3 doc_id_here
"""

import os
import sys
import json
from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # tiny / base / small / medium / large-v3 -- base is a good CPU-speed tradeoff


def normalize_audio_to_structured_content(audio_path, doc_id, model_size=MODEL_SIZE, filename=None):
    print(f"[+] Initializing audio ingestion for: {audio_path}")

    if not os.path.exists(audio_path):
        print(f"[-] Error: audio file {audio_path} not found.")
        return None

    os.makedirs("ocr_cache", exist_ok=True)

    print(f"[+] Loading faster-whisper model ({model_size}, CPU, int8)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("[+] Transcribing locally (no cloud call)...")
    segments, info = model.transcribe(audio_path, beam_size=5)

    structured_content = {
        "doc_id": doc_id,
        "filename": filename or os.path.basename(audio_path),
        "modality": "audio",
        "source_file": audio_path,
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(info.duration, 2),
        "segments": [],
    }

    full_text_parts = []
    for seg in segments:
        segment_record = {
            "segment_id": seg.id,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        }
        structured_content["segments"].append(segment_record)
        full_text_parts.append(seg.text.strip())
        print(f"    [{segment_record['start']:>7.2f}s -> {segment_record['end']:>7.2f}s] {segment_record['text']}")

    structured_content["raw_text_stream"] = " ".join(full_text_parts)

    output_cache_file = f"ocr_cache/{doc_id}_audio_structured.json"
    with open(output_cache_file, "w", encoding="utf-8") as f:
        json.dump(structured_content, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Audio ingestion complete: {len(structured_content['segments'])} segments")
    print(f"[✓] Detected language: {info.language} (confidence {info.language_probability:.3f})")
    print(f"[✓] Cached to: {output_cache_file}")

    return structured_content


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_audio.py <audio_file> <doc_id>")
        print("Example: python process_audio.py sample_meeting.mp3 meeting_01")
        sys.exit(1)

    audio_file = sys.argv[1]
    doc_id_arg = sys.argv[2]
    normalize_audio_to_structured_content(audio_file, doc_id_arg)
