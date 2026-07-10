import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline_common import float_env, int_env
from common import (
    filter_paths_by_stems,
    parse_common_args,
    selected_videos,
    work_dir_for,
    write_status,
    write_status_only,
)


def move_demucs_vocals(demucs_out: Path, vocals_dir: Path, wav: Path) -> bool:
    demucs_vocals = demucs_out / "htdemucs_ft" / wav.stem / "vocals.wav"
    out_file = vocals_dir / wav.name
    if not demucs_vocals.exists():
        return False
    shutil.move(str(demucs_vocals), str(out_file))
    print(f"[done] {out_file}")
    return True


def main() -> None:
    args = parse_common_args("V2 STEP2: separate vocals with one Demucs batch per selected chunk.")
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    audio_dir = work_dir / "audio"
    vocals_dir = work_dir / "vocals"
    demucs_out = work_dir / "demucs_raw"
    vocals_dir.mkdir(parents=True, exist_ok=True)

    stems = {video.stem for video in selected_videos(base_dir, args.chunk, args.target_stem)}
    wav_files = filter_paths_by_stems(sorted(audio_dir.glob("*.wav")), stems)
    if not wav_files:
        print(f"[error] no audio files found: {audio_dir}")
        sys.exit(1)

    recovered = 0
    for wav in wav_files:
        if (vocals_dir / wav.name).exists():
            continue
        if move_demucs_vocals(demucs_out, vocals_dir, wav):
            recovered += 1

    pending = [wav for wav in wav_files if not (vocals_dir / wav.name).exists()]
    for wav in wav_files:
        if wav not in pending:
            print(f"[skip] {wav.name} vocals exists")
    if not pending:
        print("v2 step2 done; all selected vocals already exist")
        return

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    selected_count = len(wav_files)
    pending_count = len(pending)
    skipped_count = selected_count - pending_count

    print(f"[info] selected audio files: {selected_count}")
    print(f"[info] recovered raw Demucs vocals: {recovered}")
    print(f"[info] pending audio files: {pending_count}")
    print(f"[info] skipped existing vocals: {skipped_count}")
    print(f"[info] Demucs device: {device}")
    write_status(
        base_dir,
        "V2_STEP2",
        f"{pending_count}/{selected_count} files",
        f"START separate vocals (pending={pending_count}, skipped_existing={skipped_count})",
    )
    start = time.time()
    batch_size = max(1, int_env("DEMUCS_BATCH_SIZE", 1))
    moved = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        batch_done = offset + len(batch)
        write_status_only(
            base_dir,
            "V2_STEP2",
            f"{batch_done}/{pending_count} pending",
            f"separate vocals batch {offset + 1}-{batch_done}/{pending_count}",
        )
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            "htdemucs_ft",
            "--device",
            device,
            "--shifts",
            str(int_env("DEMUCS_SHIFTS", 2)),
            "--overlap",
            str(float_env("DEMUCS_OVERLAP", 0.25)),
            "-j",
            str(int_env("DEMUCS_JOBS", 1)),
            "--out",
            str(demucs_out),
        ] + [str(wav) for wav in batch]
        result = subprocess.run(cmd, capture_output=False, env=os.environ.copy())
        batch_moved = 0
        for wav in batch:
            if move_demucs_vocals(demucs_out, vocals_dir, wav):
                moved += 1
                batch_moved += 1
        if result.returncode != 0 or batch_moved != len(batch):
            write_status(
                base_dir,
                "V2_STEP2",
                f"{moved}/{selected_count} files",
                f"FAILED separate vocals after moved={moved}; batch={offset + 1}-{batch_done}",
            )
            print("[error] Demucs failed")
            sys.exit(1)

    elapsed = time.time() - start
    write_status(
        base_dir,
        "V2_STEP2",
        f"{moved}/{selected_count} files",
        f"DONE separate vocals moved={moved}, skipped_existing={skipped_count} ({elapsed:.1f}s)",
    )
    print(f"v2 step2 done; vocals saved in: {vocals_dir}")


if __name__ == "__main__":
    main()
