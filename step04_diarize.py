import gc
import json
import os
import sys
import time
from pathlib import Path

from pipeline_common import HF_HUB_CACHE
from common import filter_paths_by_stems, parse_common_args, selected_videos, work_dir_for, write_status


DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")


def main() -> None:
    args = parse_common_args("V2 STEP4: speaker diarization for selected videos.")
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    vocals_dir = work_dir / "vocals"
    transcripts_dir = work_dir / "transcripts"
    diarized_dir = work_dir / "diarized"
    diarized_dir.mkdir(parents=True, exist_ok=True)

    stems = {video.stem for video in selected_videos(base_dir, args.chunk, args.target_stem)}
    wav_files = filter_paths_by_stems(sorted(vocals_dir.glob("*.wav")), stems)
    if not wav_files:
        print(f"[error] no vocals found in {vocals_dir}")
        sys.exit(1)

    pending = [wav for wav in wav_files if not (diarized_dir / f"{wav.stem}.json").exists()]
    for wav in wav_files:
        if wav not in pending:
            print(f"[skip] diarized output exists: {wav.stem}.json")
    if not pending:
        print("v2 step4 done; all selected diarized files already exist")
        return

    hf_token = os.environ.get("HF_TOKEN") or None
    if not hf_token:
        print("[info] HF_TOKEN is not set; trying cached pyannote files only")

    try:
        import torch
        import whisperx
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    except ModuleNotFoundError as exc:
        print(f"[error] missing diarization dependency: {exc.name}")
        print("[error] install your chosen torch/whisperx/pyannote stack, then rerun from V2_STEP4")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] loading diarization model once on {device}: {DIARIZATION_MODEL}")
    write_status(base_dir, "V2_STEP4", f"{len(pending)} files", "START speaker diarization")
    try:
        diarize_model = DiarizationPipeline(
            model_name=DIARIZATION_MODEL,
            token=hf_token,
            device=device,
            cache_dir=str(HF_HUB_CACHE),
        )
    except Exception as exc:
        print(f"[error] failed to load diarization model: {exc}")
        print("[error] check cached pyannote models and HF_TOKEN if the model is gated")
        sys.exit(1)

    start = time.time()
    try:
        for wav in pending:
            out_file = diarized_dir / f"{wav.stem}.json"
            transcript_file = transcripts_dir / f"{wav.stem}.json"
            if not transcript_file.exists():
                print(f"[skip] transcript not found: {transcript_file}")
                continue

            print(f"[process] {wav.name}")
            write_status(base_dir, "V2_STEP4", wav.name, "speaker diarization")
            with transcript_file.open(encoding="utf-8-sig") as f:
                transcript = json.load(f)

            audio = whisperx.load_audio(str(wav))
            diarize_segments = diarize_model(audio)
            result = assign_word_speakers(diarize_segments, transcript, fill_nearest=True)

            with out_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            speakers = sorted({str(seg.get("speaker", "unknown")) for seg in result.get("segments", [])})
            print(f"[done] diarized output saved: {out_file}; speakers={speakers}")
            del audio
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        gc.collect()
        if "torch" in locals() and device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - start
    write_status(base_dir, "V2_STEP4", f"{len(pending)} files", f"DONE speaker diarization ({elapsed:.1f}s)")
    print(f"v2 step4 done; diarized files saved in: {diarized_dir}")


if __name__ == "__main__":
    main()
