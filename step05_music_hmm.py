import gc
import json
import os
import subprocess
import sys
import time
import wave
from collections import Counter
from pathlib import Path

from common import StatusHeartbeat, filter_paths_by_stems, parse_common_args, selected_videos, work_dir_for, write_status, write_status_only
from override_utils import apply_music_overrides, load_overrides


WHISPER_AT_CACHE = os.environ.get("WHISPER_AT_CACHE", r"C:\Python\hf-cache\whisper-at")
MUSIC_SYMBOL = "\u266a"
WINDOW_SECONDS = 4.0
SINGING_CORE_THRESHOLD = -2.45
SINGING_SOFT_THRESHOLD = -2.70
GENERIC_SINGING_THRESHOLD = -2.10
VOCAL_MUSIC_THRESHOLD = -3.20
MUSIC_BRIDGE_GAP = 2.5
MIN_MUSIC_DURATION = 1.8
SAME_SPEAKER_BRIDGE_GAP = 3.2
BORDERLINE_SINGER_CORE = -3.05
BORDERLINE_GENERIC_SINGING = -2.65
HMM_START_MUSIC_PENALTY = 1.35
HMM_SWITCH_TO_MUSIC_PENALTY = 1.55
HMM_SWITCH_TO_SPEECH_PENALTY = 1.05
HMM_SAME_SPEAKER_STAY_BONUS = 1.15
HMM_NEAR_MUSIC_STAY_BONUS = 0.35
HMM_LONG_GAP_PENALTY = 1.40
HMM_SPEAKER_CHANGE_PENALTY = 0.35
HMM_SEED_BONUS = 1.10
HMM_AUDIO_BRIDGE_BONUS = 0.75
HMM_BORDERLINE_BONUS = 0.45
MUSIC_SPLIT_MAX_DURATION = 7.5
MUSIC_SPLIT_MAX_CHARS = 88
MUSIC_SPLIT_PAUSE_GAP = 0.55
DIALOGUE_SPLIT_MAX_DURATION = 5.5
DIALOGUE_SPLIT_MAX_CHARS = 82
DIALOGUE_SPLIT_PAUSE_GAP = 0.60
TAG_AUDIO_SAMPLE_RATE = 16000
TAG_AUDIO_CHANNELS = 1

MALE_SPEECH = 1
FEMALE_SPEECH = 2
CHILD_SPEECH = 3
CONVERSATION = 4
NARRATION = 5
SINGING = 27
MALE_SINGING = 32
FEMALE_SINGING = 33
CHILD_SINGING = 34
VOCAL_MUSIC = 254


def normalize_inline_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\\N", "\n").replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    return " ".join(parts).strip()


def normalize_music_marks(text: str, is_music: bool) -> str:
    text = normalize_inline_text(text)
    core = text.strip().strip(MUSIC_SYMBOL).strip()
    if not is_music:
        return core
    return f"{MUSIC_SYMBOL} {core} {MUSIC_SYMBOL}" if core else core


def break_text(text: str, max_chars: int) -> str:
    text = normalize_inline_text(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    break_chars = ",.!?;: " + "\uff0c\u3002\uff01\uff1f\uff1b\uff1a"
    lines: list[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = max_chars
        lower_bound = max(max_chars - 12, 1)
        for idx in range(max_chars, lower_bound - 1, -1):
            if idx < len(rest) and rest[idx] in break_chars:
                cut = idx + 1
                break
        lines.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        lines.append(rest)
    return r"\N".join(lines)


def overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def max_value(row, indexes: list[int]) -> float:
    return max(float(row[index]) for index in indexes)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def word_count(text: str) -> int:
    return len([part for part in normalize_inline_text(text).split(" ") if part])


def primary_speaker(seg: dict) -> str:
    if seg.get("speaker"):
        return str(seg.get("speaker"))
    words = seg.get("words") or []
    speakers = [str(word.get("speaker")) for word in words if word.get("speaker")]
    if not speakers:
        return ""
    return Counter(speakers).most_common(1)[0][0]


def speaker_count(seg: dict) -> int:
    speakers = set()
    if seg.get("speaker"):
        speakers.add(str(seg.get("speaker")))
    for word in seg.get("words") or []:
        if word.get("speaker"):
            speakers.add(str(word.get("speaker")))
    return len(speakers)


def window_stats(audio_tag, start: float, end: float) -> dict:
    stats = {
        "singing_peak": -999.0,
        "singer_core_peak": -999.0,
        "speech_peak": -999.0,
        "conversation_peak": -999.0,
        "vocal_music_peak": -999.0,
        "overlap": 0.0,
    }
    total = 0.0
    sums = {
        "singing_mean": 0.0,
        "singer_core_mean": 0.0,
        "speech_mean": 0.0,
        "conversation_mean": 0.0,
        "vocal_music_mean": 0.0,
    }
    for index in range(audio_tag.shape[0]):
        win_start = index * WINDOW_SECONDS
        win_end = win_start + WINDOW_SECONDS
        weight = overlap(start, end, win_start, win_end)
        if weight <= 0:
            continue
        row = audio_tag[index]
        singing = max_value(row, [SINGING, MALE_SINGING, FEMALE_SINGING, CHILD_SINGING])
        singer_core = max_value(row, [MALE_SINGING, FEMALE_SINGING, CHILD_SINGING])
        speech = max_value(row, [MALE_SPEECH, FEMALE_SPEECH, CHILD_SPEECH])
        conversation = max_value(row, [CONVERSATION, NARRATION])
        vocal_music = float(row[VOCAL_MUSIC])

        stats["singing_peak"] = max(stats["singing_peak"], singing)
        stats["singer_core_peak"] = max(stats["singer_core_peak"], singer_core)
        stats["speech_peak"] = max(stats["speech_peak"], speech)
        stats["conversation_peak"] = max(stats["conversation_peak"], conversation)
        stats["vocal_music_peak"] = max(stats["vocal_music_peak"], vocal_music)
        stats["overlap"] += weight
        total += weight

        sums["singing_mean"] += singing * weight
        sums["singer_core_mean"] += singer_core * weight
        sums["speech_mean"] += speech * weight
        sums["conversation_mean"] += conversation * weight
        sums["vocal_music_mean"] += vocal_music * weight

    if total > 0:
        for key, value in sums.items():
            stats[key] = value / total
    else:
        for key in sums:
            stats[key] = -999.0
    return stats


def looks_like_short_dialogue(seg: dict) -> bool:
    text = normalize_inline_text(seg.get("text", ""))
    duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    words = word_count(text)
    if not text:
        return True
    if text.endswith("?"):
        return True
    if duration < 1.0:
        return True
    if duration < 2.2 and words <= 4:
        return True
    if text.endswith("!") and duration < 2.4:
        return True
    return False


def seed_music(seg: dict, stats: dict) -> tuple[bool, str]:
    duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    if duration < MIN_MUSIC_DURATION:
        return False, "too_short"
    if looks_like_short_dialogue(seg):
        return False, "short_dialogue"
    if stats["conversation_peak"] > -4.2:
        return False, "conversation_peak"
    if stats["speech_peak"] > -2.3 and stats["singer_core_peak"] < -1.75:
        return False, "speech_peak"

    strong_core = stats["singer_core_peak"] >= SINGING_CORE_THRESHOLD
    strong_generic = stats["singing_peak"] >= GENERIC_SINGING_THRESHOLD and stats["singer_core_peak"] >= -3.0
    soft_core = stats["singer_core_mean"] >= SINGING_SOFT_THRESHOLD
    vocal_support = stats["vocal_music_peak"] >= VOCAL_MUSIC_THRESHOLD
    low_dialogue = stats["conversation_peak"] <= -5.0 and stats["speech_peak"] <= -3.4
    if strong_core:
        return True, "strong_singer_core"
    if strong_generic:
        return True, "strong_generic_singing"
    if soft_core and vocal_support and low_dialogue:
        return True, "soft_core_with_vocal_support"
    return False, "below_threshold"


def can_bridge(seg: dict, stats: dict) -> bool:
    duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    if duration < 2.0:
        return False
    if looks_like_short_dialogue(seg):
        return False
    if stats["conversation_peak"] > -4.4:
        return False
    return stats["singer_core_peak"] >= -2.95 or stats["singer_core_mean"] >= -3.0


def speaker_borderline_music(seg: dict, stats: dict) -> bool:
    if looks_like_short_dialogue(seg):
        return False
    if speaker_count(seg) > 2:
        return False
    if stats["conversation_peak"] > -4.25:
        return False
    if stats["speech_peak"] > -2.15 and stats["singer_core_peak"] < -2.15:
        return False
    return (
        stats["singer_core_peak"] >= BORDERLINE_SINGER_CORE
        or stats["singing_peak"] >= BORDERLINE_GENERIC_SINGING
        or (stats["singer_core_mean"] >= -3.15 and stats["vocal_music_peak"] >= VOCAL_MUSIC_THRESHOLD)
    )


def zero_peak_stats(stats: dict) -> bool:
    keys = [
        "singing_peak",
        "singer_core_peak",
        "speech_peak",
        "conversation_peak",
        "vocal_music_peak",
    ]
    return all(abs(float(stats.get(key, 0.0))) < 0.0001 for key in keys)


def wav_audio_params(path: Path) -> tuple[int, int] | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getframerate(), handle.getnchannels()
    except Exception:
        return None


def whisper_at_audio_file(wav: Path, out_dir: Path, base_dir: Path | None = None) -> Path:
    params = wav_audio_params(wav)
    if params == (TAG_AUDIO_SAMPLE_RATE, TAG_AUDIO_CHANNELS):
        return wav

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / wav.name
    if out_file.exists() and out_file.stat().st_size > 0 and out_file.stat().st_mtime >= wav.stat().st_mtime:
        return out_file

    if base_dir is not None:
        write_status_only(base_dir, "V2_STEP5", wav.name, "normalizing audio for Whisper-AT")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(TAG_AUDIO_SAMPLE_RATE),
        "-ac",
        str(TAG_AUDIO_CHANNELS),
        str(out_file),
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0 or not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError(f"failed to normalize Whisper-AT audio: {wav}")
    return out_file


def dialogue_phrase_penalty(seg: dict) -> float:
    text = normalize_inline_text(seg.get("text", ""))
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return 2.5
    short_dialogue = {
        "hi",
        "hey",
        "yeah",
        "yes",
        "no",
        "okay",
        "ok",
        "thanks",
        "thank you",
        "thank you very much",
        "all right",
        "alright",
        "sorry",
        "i m sorry",
    }
    if cleaned in short_dialogue:
        return 2.4
    if cleaned.startswith("thank you") and word_count(text) <= 5:
        return 2.2
    return 0.0


def hmm_observation_score(seg: dict, stats: dict, seed_flag: bool, bridge_flag: bool) -> float:
    duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    text = normalize_inline_text(seg.get("text", ""))
    words = word_count(text)

    if zero_peak_stats(stats):
        score = -0.45
        score += clamp((stats["singer_core_mean"] - SINGING_SOFT_THRESHOLD) / 0.85, -2.5, 2.5) * 1.40
        score += clamp((stats["singing_mean"] - SINGING_SOFT_THRESHOLD) / 0.85, -2.0, 2.0) * 0.70
        score += clamp((stats["vocal_music_mean"] - VOCAL_MUSIC_THRESHOLD) / 1.20, -1.5, 1.5) * 0.35
        if stats["conversation_mean"] > -4.20:
            score -= clamp((stats["conversation_mean"] + 4.20) / 1.0, 0.0, 4.0) * 1.10
        if stats["speech_mean"] > -2.60:
            score -= clamp((stats["speech_mean"] + 2.60) / 0.90, 0.0, 3.5) * 0.85
        if duration >= 3.0 and words >= 6 and stats["conversation_mean"] <= -3.80 and stats["speech_mean"] <= -3.00:
            score += 0.40
        if duration >= 8.0 and words >= 8 and stats["conversation_mean"] <= -4.20 and stats["speech_mean"] <= -3.20:
            score += 0.60
        if duration < MIN_MUSIC_DURATION:
            score -= 3.25
        if text.endswith("?"):
            score -= 3.0
        if words <= 2:
            score -= 2.4
        elif words <= 3:
            score -= 1.7
        elif words <= 4 and duration < 3.2:
            score -= 1.0
        score -= dialogue_phrase_penalty(seg)
        if speaker_count(seg) > 2:
            score -= 0.8
        if bridge_flag:
            score += HMM_AUDIO_BRIDGE_BONUS
        return score

    score = -0.35
    score += clamp((stats["singer_core_peak"] - SINGING_CORE_THRESHOLD) / 0.70, -3.0, 3.0) * 1.70
    score += clamp((stats["singing_peak"] - GENERIC_SINGING_THRESHOLD) / 0.80, -2.5, 2.5) * 0.85
    score += clamp((stats["singer_core_mean"] - SINGING_SOFT_THRESHOLD) / 0.85, -2.5, 2.5) * 0.95
    score += clamp((stats["vocal_music_peak"] - VOCAL_MUSIC_THRESHOLD) / 1.20, -1.5, 1.5) * 0.35

    strong_singing = stats["singer_core_peak"] >= -1.75 or stats["singing_peak"] >= -1.65
    dialogue_penalty_scale = 0.45 if strong_singing else 1.0
    if stats["conversation_peak"] > -4.25:
        score -= clamp((stats["conversation_peak"] + 4.25) / 1.0, 0.0, 4.0) * 1.05 * dialogue_penalty_scale
    if stats["speech_peak"] > -2.30:
        score -= clamp((stats["speech_peak"] + 2.30) / 0.90, 0.0, 3.5) * 0.80 * dialogue_penalty_scale
    if stats["conversation_mean"] > -4.40:
        score -= clamp((stats["conversation_mean"] + 4.40) / 1.2, 0.0, 2.0) * 0.45 * dialogue_penalty_scale

    if duration < MIN_MUSIC_DURATION:
        score -= 3.25
    if text.endswith("?"):
        score -= 3.0
    if words <= 2:
        score -= 2.4
    elif words <= 3:
        score -= 1.7
    elif words <= 4 and duration < 3.2:
        score -= 1.0
    score -= dialogue_phrase_penalty(seg)

    if speaker_count(seg) > 2:
        score -= 0.8
    if seed_flag:
        score += HMM_SEED_BONUS
    if bridge_flag:
        score += HMM_AUDIO_BRIDGE_BONUS
    if speaker_borderline_music(seg, stats) or can_bridge(seg, stats):
        score += HMM_BORDERLINE_BONUS

    return score


def hmm_transition_score(prev_state: int, curr_state: int, prev_seg: dict, curr_seg: dict) -> float:
    if prev_state == 0 and curr_state == 0:
        return 0.0

    gap = max(0.0, float(curr_seg.get("start", 0.0)) - float(prev_seg.get("end", 0.0)))
    prev_speaker = primary_speaker(prev_seg)
    curr_speaker = primary_speaker(curr_seg)
    same_speaker = bool(prev_speaker and curr_speaker and prev_speaker == curr_speaker)
    speaker_changed = bool(prev_speaker and curr_speaker and prev_speaker != curr_speaker)

    if prev_state == 0 and curr_state == 1:
        penalty = HMM_SWITCH_TO_MUSIC_PENALTY
        if gap <= MUSIC_BRIDGE_GAP:
            penalty -= 0.15
        return -penalty

    if prev_state == 1 and curr_state == 0:
        penalty = HMM_SWITCH_TO_SPEECH_PENALTY
        if gap > SAME_SPEAKER_BRIDGE_GAP:
            penalty -= 0.15
        return -penalty

    score = 0.0
    if gap <= SAME_SPEAKER_BRIDGE_GAP and same_speaker:
        score += HMM_SAME_SPEAKER_STAY_BONUS
    elif gap <= MUSIC_BRIDGE_GAP:
        score += HMM_NEAR_MUSIC_STAY_BONUS
    elif gap > 8.0:
        score -= HMM_LONG_GAP_PENALTY
    else:
        score -= 0.45

    if speaker_changed:
        score -= HMM_SPEAKER_CHANGE_PENALTY
    return score


def viterbi_music_flags(segments: list[dict], scores: list[float]) -> list[bool]:
    if not segments:
        return []

    neg_inf = -1_000_000.0
    dp: list[list[float]] = [[0.0, scores[0] - HMM_START_MUSIC_PENALTY]]
    back: list[list[int]] = [[0, 0]]

    for index in range(1, len(segments)):
        curr_dp = [neg_inf, neg_inf]
        curr_back = [0, 0]
        for curr_state in (0, 1):
            emit = scores[index] if curr_state == 1 else 0.0
            for prev_state in (0, 1):
                value = (
                    dp[index - 1][prev_state]
                    + hmm_transition_score(prev_state, curr_state, segments[index - 1], segments[index])
                    + emit
                )
                if value > curr_dp[curr_state]:
                    curr_dp[curr_state] = value
                    curr_back[curr_state] = prev_state
        dp.append(curr_dp)
        back.append(curr_back)

    state = 1 if dp[-1][1] > dp[-1][0] else 0
    flags = [False] * len(segments)
    for index in range(len(segments) - 1, -1, -1):
        flags[index] = state == 1
        state = back[index][state]
    return flags


def same_speaker_music_context(index: int, segments: list[dict], flags: list[bool]) -> bool:
    if index <= 0 or index >= len(segments) - 1:
        return False
    speaker = primary_speaker(segments[index])
    if not speaker:
        return False
    prev_seg = segments[index - 1]
    next_seg = segments[index + 1]
    prev_gap = float(segments[index].get("start", 0.0)) - float(prev_seg.get("end", 0.0))
    next_gap = float(next_seg.get("start", 0.0)) - float(segments[index].get("end", 0.0))
    return (
        flags[index - 1]
        and flags[index + 1]
        and speaker == primary_speaker(prev_seg) == primary_speaker(next_seg)
        and prev_gap <= SAME_SPEAKER_BRIDGE_GAP
        and next_gap <= SAME_SPEAKER_BRIDGE_GAP
    )


def hmm_reason(
    index: int,
    segments: list[dict],
    flags: list[bool],
    seed_flags: list[bool],
    seed_reasons: list[str],
    bridge_flags: list[bool],
    scores: list[float],
) -> str:
    if not flags[index]:
        if seed_flags[index]:
            return f"hmm_rejected:{seed_reasons[index]}"
        return seed_reasons[index]
    if seed_flags[index]:
        return seed_reasons[index]
    if bridge_flags[index]:
        return "hmm_audio_bridge"
    if same_speaker_music_context(index, segments, flags):
        return "hmm_speaker_continuity"
    if scores[index] >= 2.0:
        return "hmm_audio_score"
    return "hmm_viterbi"


def music_tail_candidate(prev_seg: dict, seg: dict, stats: dict) -> bool:
    gap = float(seg.get("start", 0.0)) - float(prev_seg.get("end", 0.0))
    if gap < -0.1 or gap > 1.6:
        return False
    if primary_speaker(prev_seg) != primary_speaker(seg):
        return False
    if word_count(seg.get("text", "")) < 4:
        return False
    if looks_like_short_dialogue(seg):
        return False
    if stats["speech_peak"] > -2.30:
        return False
    if stats["conversation_mean"] > -3.35:
        return False
    return stats["singing_peak"] >= -2.20 or stats["singer_core_peak"] >= -2.20


def word_start(word: dict, fallback: float) -> float:
    try:
        return float(word.get("start", fallback))
    except (TypeError, ValueError):
        return fallback


def word_end(word: dict, fallback: float) -> float:
    try:
        return float(word.get("end", fallback))
    except (TypeError, ValueError):
        return fallback


def valid_timed_words(words: list[dict]) -> bool:
    timed = 0
    for word in words:
        if word.get("start") is not None and word.get("end") is not None:
            timed += 1
    return timed >= 2


def words_text(words: list[dict]) -> str:
    return normalize_inline_text(" ".join(str(word.get("word", "")).strip() for word in words if str(word.get("word", "")).strip()))


def split_profile(seg: dict) -> dict:
    if seg.get("is_music"):
        return {
            "name": "music",
            "max_duration": MUSIC_SPLIT_MAX_DURATION,
            "max_chars": MUSIC_SPLIT_MAX_CHARS,
            "pause_gap": MUSIC_SPLIT_PAUSE_GAP,
            "min_duration": 1.5,
            "min_words": 4,
            "lookahead": 6,
        }
    return {
        "name": "dialogue",
        "max_duration": DIALOGUE_SPLIT_MAX_DURATION,
        "max_chars": DIALOGUE_SPLIT_MAX_CHARS,
        "pause_gap": DIALOGUE_SPLIT_PAUSE_GAP,
        "min_duration": 1.2,
        "min_words": 3,
        "lookahead": 7,
    }


def segment_needs_split(seg: dict, profile: dict) -> bool:
    duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    text = normalize_inline_text(seg.get("text", ""))
    return duration > profile["max_duration"] or len(text) > profile["max_chars"]


def boundary_score(words: list[dict], start: int, split_at: int, profile: dict) -> float:
    prev = words[split_at - 1]
    nxt = words[split_at]
    chunk_start = word_start(words[start], 0.0)
    chunk_end = word_end(prev, chunk_start)
    duration = max(0.0, chunk_end - chunk_start)
    text = words_text(words[start:split_at])
    gap = max(0.0, word_start(nxt, chunk_end) - chunk_end)
    token = str(prev.get("word", "")).strip()
    prev_clean = "".join(ch.lower() for ch in token if ch.isalnum())
    next_clean = "".join(ch.lower() for ch in str(nxt.get("word", "")).strip() if ch.isalnum())

    score = -abs(duration - profile["max_duration"]) / max(profile["max_duration"], 0.1)
    if (prev_clean, next_clean) in {
        ("all", "right"),
        ("you", "know"),
        ("i", "mean"),
        ("kind", "of"),
        ("sort", "of"),
        ("a", "lot"),
        ("going", "to"),
        ("want", "to"),
        ("got", "to"),
        ("have", "to"),
    }:
        score -= 8.0
    if token.endswith((".", "?", "!", ";", ":", "\u3002", "\uff1f", "\uff01", "\uff1b", "\uff1a")):
        score += 4.0
    elif token.endswith((",", "\uff0c")):
        score += 2.0
    if gap >= profile["pause_gap"]:
        score += 3.0 + min(gap, 1.5)
    if prev.get("speaker") and nxt.get("speaker") and prev.get("speaker") != nxt.get("speaker"):
        score += 5.0
    if duration >= profile["max_duration"] or len(text) >= profile["max_chars"]:
        score += 1.0
    if duration < profile["min_duration"] or len(text.split()) < profile["min_words"]:
        score -= 4.0
    return score


def choose_split_index(words: list[dict], start: int, profile: dict) -> int | None:
    min_words = profile["min_words"]
    if len(words) - start <= min_words * 2:
        return None

    limit = None
    for index in range(start + min_words, len(words) - min_words + 1):
        text = words_text(words[start:index])
        duration = word_end(words[index - 1], word_start(words[start], 0.0)) - word_start(words[start], 0.0)
        if duration >= profile["max_duration"] or len(text) >= profile["max_chars"]:
            limit = index
            break
    if limit is None:
        full_text = words_text(words[start:])
        full_duration = word_end(words[-1], word_start(words[start], 0.0)) - word_start(words[start], 0.0)
        if full_duration < profile["max_duration"] and len(full_text) < profile["max_chars"]:
            return None
        limit = max(start + min_words, len(words) - min_words)

    search_end = min(len(words) - min_words, limit + profile["lookahead"])
    best_index = None
    best_score = -999.0
    for index in range(start + min_words, search_end + 1):
        chunk_start = word_start(words[start], 0.0)
        chunk_end = word_end(words[index - 1], chunk_start)
        remaining_start = word_start(words[index], chunk_end)
        if chunk_end - chunk_start < profile["min_duration"]:
            continue
        if word_end(words[-1], remaining_start) - remaining_start < profile["min_duration"]:
            continue
        score = boundary_score(words, start, index, profile)
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is not None:
        return best_index
    return limit if start + min_words <= limit <= len(words) - min_words else None


def majority_speaker(words: list[dict], fallback: str) -> str:
    speakers = [str(word.get("speaker")) for word in words if word.get("speaker")]
    if not speakers:
        return fallback
    return Counter(speakers).most_common(1)[0][0]


def split_segment(seg: dict) -> list[dict]:
    words = [word for word in (seg.get("words") or []) if isinstance(word, dict)]
    profile = split_profile(seg)
    if not segment_needs_split(seg, profile) or not valid_timed_words(words):
        return [seg]

    boundaries = [0]
    start = 0
    while True:
        split_at = choose_split_index(words, start, profile)
        if split_at is None:
            break
        boundaries.append(split_at)
        start = split_at
        if len(words) - start <= profile["min_words"]:
            break
    boundaries.append(len(words))
    if len(boundaries) <= 2:
        return [seg]

    parts: list[dict] = []
    original_start = float(seg.get("start", 0.0))
    original_end = float(seg.get("end", original_start))
    fallback_speaker = primary_speaker(seg)
    for part_index, (start_idx, end_idx) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        part_words = [dict(word) for word in words[start_idx:end_idx]]
        if not part_words:
            continue
        part = dict(seg)
        part["words"] = part_words
        part["start"] = round(word_start(part_words[0], original_start), 3)
        part["end"] = round(word_end(part_words[-1], part["start"]), 3)
        part["text"] = words_text(part_words)
        part["speaker"] = majority_speaker(part_words, fallback_speaker)
        part["segment_refined"] = True
        part["split_profile"] = profile["name"]
        part["split_from"] = {"start": round(original_start, 3), "end": round(original_end, 3)}
        part["split_index"] = part_index
        part["split_count"] = len(boundaries) - 1
        if part.get("is_music") and part.get("music_reason") and not str(part["music_reason"]).endswith(":split"):
            part["music_reason"] = f"{part['music_reason']}:split"
        parts.append(part)
    return parts or [seg]


def refine_segments(segments: list[dict]) -> tuple[list[dict], int]:
    refined: list[dict] = []
    split_count = 0
    for seg in segments:
        parts = split_segment(seg)
        if len(parts) > 1:
            split_count += len(parts) - 1
        refined.extend(parts)
    return refined, split_count


def update_music_status(base_dir: Path, video: str, phase: str, index: int, total: int) -> None:
    progress = 100.0 if total <= 0 else index * 100.0 / total
    write_status_only(base_dir, "V2_STEP5", video, f"{phase} {index}/{total} ({progress:.1f}%)")


def apply_music_flags(segments: list[dict], audio_tag, base_dir: Path | None = None, video: str = "") -> int:
    total = max(len(segments), 1)
    if base_dir is not None:
        update_music_status(base_dir, video, "music smoothing stats", 0, total)
    stats_list = [
        window_stats(audio_tag, float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
        for seg in segments
    ]
    if base_dir is not None:
        update_music_status(base_dir, video, "music smoothing seeds", min(len(segments), total), total)
    seed_results = [seed_music(seg, stats) for seg, stats in zip(segments, stats_list)]
    seed_flags = [flag for flag, _reason in seed_results]
    seed_reasons = [reason for _flag, reason in seed_results]
    bridge_flags = [False] * len(segments)

    for index in range(1, len(segments) - 1):
        if seed_flags[index]:
            continue
        prev = segments[index - 1]
        curr = segments[index]
        nxt = segments[index + 1]
        prev_gap = float(curr.get("start", 0.0)) - float(prev.get("end", 0.0))
        next_gap = float(nxt.get("start", 0.0)) - float(curr.get("end", 0.0))
        if seed_flags[index - 1] and seed_flags[index + 1] and prev_gap <= MUSIC_BRIDGE_GAP and next_gap <= MUSIC_BRIDGE_GAP:
            if can_bridge(curr, stats_list[index]):
                bridge_flags[index] = True
        if base_dir is not None and index % 100 == 0:
            update_music_status(base_dir, video, "music smoothing audio bridge", index, total)

    if base_dir is not None:
        update_music_status(base_dir, video, "music smoothing HMM", 0, total)
    hmm_scores = [
        hmm_observation_score(seg, stats, seed_flag, bridge_flag)
        for seg, stats, seed_flag, bridge_flag in zip(segments, stats_list, seed_flags, bridge_flags)
    ]
    flags = viterbi_music_flags(segments, hmm_scores)
    tail_flags = [False] * len(segments)
    for index in range(1, len(segments)):
        if flags[index] or not flags[index - 1]:
            continue
        if music_tail_candidate(segments[index - 1], segments[index], stats_list[index]):
            flags[index] = True
            tail_flags[index] = True
    reasons = [
        "hmm_music_tail"
        if tail_flags[index]
        else hmm_reason(index, segments, flags, seed_flags, seed_reasons, bridge_flags, hmm_scores)
        for index in range(len(segments))
    ]
    if base_dir is not None:
        update_music_status(base_dir, video, "music smoothing HMM", total, total)

    changed = 0
    for index, (seg, is_music, reason, stats, hmm_score) in enumerate(
        zip(segments, flags, reasons, stats_list, hmm_scores),
        start=1,
    ):
        if bool(seg.get("is_music", False)) != bool(is_music):
            changed += 1
        seg["is_music"] = bool(is_music)
        seg["music_reason"] = reason if is_music else f"not_music:{reason}"
        seg["music_stats"] = {
            key: round(float(value), 4)
            for key, value in stats.items()
            if key.endswith("_peak") or key.endswith("_mean")
        }
        seg["music_stats"]["hmm_score"] = round(float(hmm_score), 4)
        if base_dir is not None and (index == total or index % 100 == 0):
            update_music_status(base_dir, video, "music smoothing write flags", index, total)
    return changed


def main() -> None:
    args = parse_common_args("V2 STEP5: Whisper-AT HMM music tagging.")
    base_dir = Path(args.base_dir).resolve()
    work_dir = work_dir_for(base_dir, args.work_dir)
    vocals_dir = work_dir / "vocals"
    diarized_dir = work_dir / "diarized"
    marked_dir = work_dir / "music_marked"
    cache_dir = work_dir / "music_tags"
    tag_audio_dir = work_dir / "music_audio"
    marked_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_audio_dir.mkdir(parents=True, exist_ok=True)

    stems = {video.stem for video in selected_videos(base_dir, args.chunk, args.target_stem)}
    wav_files = filter_paths_by_stems(sorted(vocals_dir.glob("*.wav")), stems)
    if not wav_files:
        print(f"[error] no vocals found in {vocals_dir}")
        sys.exit(1)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Whisper-AT device: {device}")
    write_status(base_dir, "V2_STEP5", f"{len(wav_files)} files", "START music HMM tagging")
    start = time.time()
    model = None

    try:
        for wav in wav_files:
            json_file = diarized_dir / f"{wav.stem}.json"
            out_file = marked_dir / f"{wav.stem}.json"
            cache_file = cache_dir / f"{wav.stem}.pt"
            if not json_file.exists():
                print(f"[skip] diarized subtitle not found: {json_file.name}")
                continue

            print(f"[process] {wav.name}")
            write_status(base_dir, "V2_STEP5", wav.name, "Whisper-AT music HMM tagging")
            if cache_file.exists():
                write_status_only(base_dir, "V2_STEP5", wav.name, "loading cached Whisper-AT tags")
                result = torch.load(cache_file, map_location="cpu")
            else:
                if model is None:
                    import whisper_at

                    print("[info] loading Whisper-AT large-v2")
                    model = whisper_at.load_model("large-v2", device=device, download_root=WHISPER_AT_CACHE)
                tag_wav = whisper_at_audio_file(wav, tag_audio_dir, base_dir)
                if tag_wav != wav:
                    print(f"[audio] Whisper-AT normalized input: {tag_wav.name}")
                with StatusHeartbeat(base_dir, "V2_STEP5", wav.name, "running Whisper-AT audio tagging"):
                    result = whisper_at.transcribe(
                        model,
                        str(tag_wav),
                        at_time_res=WINDOW_SECONDS,
                        language="en",
                        verbose=False,
                        fp16=torch.cuda.is_available(),
                    )
                torch.save(
                    {
                        "audio_tag": result["audio_tag"].cpu(),
                        "at_time_res": result["at_time_res"],
                    },
                    cache_file,
                )

            with json_file.open(encoding="utf-8-sig") as f:
                data = json.load(f)

            segments = data.get("segments", [])
            before_segments = len(segments)
            changed = apply_music_flags(segments, result["audio_tag"], base_dir, wav.name)
            override_count = apply_music_overrides(segments, load_overrides(base_dir, wav.stem))
            data["segments"] = segments
            music_count = sum(1 for seg in segments if seg.get("is_music"))

            with out_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(
                f"[done] {out_file.name}; segments={before_segments}->{len(segments)}; "
                f"music_segments={music_count}; changed={changed}; overrides={override_count}"
            )
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - start
    write_status(base_dir, "V2_STEP5", f"{len(wav_files)} files", f"DONE music HMM tagging ({elapsed:.1f}s)")
    print(f"v2 step5 done; music-marked files saved in: {marked_dir}")


if __name__ == "__main__":
    main()
