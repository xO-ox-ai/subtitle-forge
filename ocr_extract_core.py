import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from pipeline_common import FFMPEG_DIR, PADDLEOCR_SCRIPTS, exe_path, float_env, int_env


def _default_ocr_max_frames(interval_seconds: float) -> int:
    return max(1800, int(3600 / interval_seconds) + 1)


OCR_INTERVAL_SECONDS = max(0.5, float_env("OCR_INTERVAL_SECONDS", 0.5))
OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.45"))
OCR_MAX_FRAMES = int_env("OCR_MAX_FRAMES", _default_ocr_max_frames(OCR_INTERVAL_SECONDS))
OCR_MAX_MERGED_SECONDS = float_env("OCR_MAX_MERGED_SECONDS", 6.0)
OCR_MAX_MERGE_CENTER_JUMP_PIXELS = float_env("OCR_MAX_MERGE_CENTER_JUMP_PIXELS", 420.0)
OCR_LANG = os.environ.get("OCR_LANG", "en")
PADDLEOCR_DEVICE = os.environ.get("PADDLEOCR_DEVICE", "gpu")


def configure_interval(interval_seconds: float) -> None:
    global OCR_INTERVAL_SECONDS, OCR_MAX_FRAMES
    OCR_INTERVAL_SECONDS = max(0.5, float(interval_seconds))
    OCR_MAX_FRAMES = int_env("OCR_MAX_FRAMES", _default_ocr_max_frames(OCR_INTERVAL_SECONDS))


def write_status(base_dir: Path, video: str, index: int, total: int) -> None:
    return None


def ocr_temporary_directory(out_file: Path) -> tempfile.TemporaryDirectory:
    """Create OCR scratch space inside the configured pipeline work directory."""
    work_dir = out_file.parent.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="sub_ocr_", dir=work_dir)


def run_json(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def video_info(video: Path) -> tuple[float, int, int]:
    ffprobe = exe_path("ffprobe.exe", "FFPROBE_EXE", (FFMPEG_DIR,))
    data = run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(video),
        ]
    )
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    width = int(stream.get("width") or 1920)
    height = int(stream.get("height") or 1080)
    return duration, width, height


def sample_times(duration: float) -> list[float]:
    if duration <= 0:
        return []
    times: list[float] = []
    current = OCR_INTERVAL_SECONDS / 2
    while current < duration and len(times) < OCR_MAX_FRAMES:
        times.append(round(current, 3))
        current += OCR_INTERVAL_SECONDS
    return times


def extract_frame(video: Path, timestamp: float, out_file: Path) -> bool:
    ffmpeg = exe_path("ffmpeg.exe", "FFMPEG_EXE", (FFMPEG_DIR,))
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_file),
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def box_from_points(points) -> list[float]:
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return []


def box_from_value(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 4 and all(not isinstance(item, (list, tuple)) for item in value[:4]):
            try:
                x1, y1, x2, y2 = [float(item) for item in value[:4]]
                return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            except (TypeError, ValueError):
                return []
        return box_from_points(value)
    return []


def env_flag(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return None
    return str(value)


class PaddleOCRCLIReader:
    def __init__(self) -> None:
        self.exe = exe_path("paddleocr.exe", "PADDLEOCR_EXE", (PADDLEOCR_SCRIPTS,))
        if not shutil.which(self.exe) and not Path(self.exe).exists():
            raise RuntimeError("paddleocr.exe not found")

    def command(self, input_dir: Path, output_dir: Path) -> list[str]:
        cmd = [
            self.exe,
            "ocr",
            "--input",
            str(input_dir),
            "--device",
            PADDLEOCR_DEVICE,
            "--save_path",
            str(output_dir),
            "--lang",
            OCR_LANG,
            "--ocr_version",
            os.environ.get("PADDLEOCR_VERSION", "PP-OCRv6"),
            "--text_rec_score_thresh",
            str(OCR_MIN_CONFIDENCE),
        ]
        optional_flags = {
            "PADDLEOCR_ENABLE_HPI": "--enable_hpi",
            "PADDLEOCR_USE_TENSORRT": "--use_tensorrt",
            "PADDLEOCR_PRECISION": "--precision",
            "PADDLEOCR_ENGINE": "--engine",
            "PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY": "--use_doc_orientation_classify",
            "PADDLEOCR_USE_DOC_UNWARPING": "--use_doc_unwarping",
            "PADDLEOCR_USE_TEXTLINE_ORIENTATION": "--use_textline_orientation",
            "PADDLEOCR_TEXT_DET_LIMIT_SIDE_LEN": "--text_det_limit_side_len",
            "PADDLEOCR_TEXT_DET_LIMIT_TYPE": "--text_det_limit_type",
            "PADDLEOCR_TEXT_DET_THRESH": "--text_det_thresh",
            "PADDLEOCR_TEXT_DET_BOX_THRESH": "--text_det_box_thresh",
            "PADDLEOCR_TEXT_DET_UNCLIP_RATIO": "--text_det_unclip_ratio",
            "PADDLEOCR_TEXT_RECOGNITION_BATCH_SIZE": "--text_recognition_batch_size",
        }
        for env_name, flag in optional_flags.items():
            value = env_flag(env_name)
            if value is not None:
                cmd.extend([flag, value])
        extra = os.environ.get("PADDLEOCR_EXTRA_ARGS")
        if extra:
            cmd.extend(shlex.split(extra, posix=False))
        return cmd

    def parse_result(self, path: Path) -> list[dict]:
        try:
            with path.open(encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return []
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        boxes = data.get("rec_boxes") or data.get("rec_polys") or data.get("dt_polys") or []
        items = []
        for index, text in enumerate(texts):
            clean = normalize_text(text)
            if not clean:
                continue
            try:
                conf = float(scores[index]) if index < len(scores) else 1.0
            except (TypeError, ValueError):
                conf = 1.0
            if conf < OCR_MIN_CONFIDENCE:
                continue
            bbox = box_from_value(boxes[index]) if index < len(boxes) else []
            items.append({"text": clean, "confidence": conf, "bbox": bbox})
        return items

    def read_many(self, images: list[Path]) -> dict[Path, list[dict]]:
        if not images:
            return {}
        input_dir = images[0].parent
        output_dir = input_dir.parent / "paddleocr_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.command(input_dir, output_dir)
        print("[ocr] " + " ".join(f'"{part}"' if " " in part else part for part in cmd))
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            raise RuntimeError("paddleocr CLI failed")
        output: dict[Path, list[dict]] = {}
        for image in images:
            result_file = output_dir / f"{image.stem}_res.json"
            output[image] = self.parse_result(result_file)
        return output

    def read(self, image: Path) -> list[dict]:
        return self.read_many([image]).get(image, [])


class PaddleReader:
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self.reader = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, show_log=False)

    def read(self, image: Path) -> list[dict]:
        result = self.reader.ocr(str(image), cls=True)
        rows = result[0] if result and isinstance(result, list) else []
        items = []
        for row in rows or []:
            if not row or len(row) < 2:
                continue
            bbox, rec = row[0], row[1]
            text = normalize_text(rec[0] if rec else "")
            conf = float(rec[1] if len(rec) > 1 else 1.0)
            if text and conf >= OCR_MIN_CONFIDENCE:
                items.append({"text": text, "confidence": conf, "bbox": box_from_points(bbox)})
        return items


class EasyOCRReader:
    def __init__(self) -> None:
        import easyocr

        langs = [part.strip() for part in OCR_LANG.split(",") if part.strip()] or ["en"]
        self.reader = easyocr.Reader(langs, gpu=False)

    def read(self, image: Path) -> list[dict]:
        rows = self.reader.readtext(str(image))
        items = []
        for bbox, text, conf in rows:
            text = normalize_text(text)
            conf = float(conf)
            if text and conf >= OCR_MIN_CONFIDENCE:
                items.append({"text": text, "confidence": conf, "bbox": box_from_points(bbox)})
        return items


class TesseractReader:
    def __init__(self) -> None:
        exe = shutil.which("tesseract.exe") or shutil.which("tesseract")
        if not exe:
            raise RuntimeError("tesseract not found")
        self.exe = exe

    def read(self, image: Path) -> list[dict]:
        cmd = [self.exe, str(image), "stdout", "-l", OCR_LANG, "--psm", "6", "tsv"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            return []
        items = []
        for line in result.stdout.splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) < 12:
                continue
            text = normalize_text(cols[11])
            if not text:
                continue
            try:
                conf = float(cols[10]) / 100.0
                x, y, w, h = [float(cols[index]) for index in (6, 7, 8, 9)]
            except ValueError:
                continue
            if conf >= OCR_MIN_CONFIDENCE:
                items.append({"text": text, "confidence": conf, "bbox": [x, y, x + w, y + h]})
        return items


def make_reader():
    errors = []
    for cls in (PaddleOCRCLIReader, PaddleReader, EasyOCRReader, TesseractReader):
        try:
            return cls()
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("no OCR engine available; tried " + "; ".join(errors))


def similar_text(left: str, right: str) -> bool:
    lkey = re.sub(r"[^a-z0-9]+", "", left.lower())
    rkey = re.sub(r"[^a-z0-9]+", "", right.lower())
    return bool(lkey and rkey and (lkey == rkey or lkey in rkey or rkey in lkey))


def bbox_center(bbox) -> tuple[float, float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_jumped(left_bbox, right_bbox) -> bool:
    left = bbox_center(left_bbox)
    right = bbox_center(right_bbox)
    if left is None or right is None:
        return False
    return math.hypot(left[0] - right[0], left[1] - right[1]) > OCR_MAX_MERGE_CENTER_JUMP_PIXELS


def can_merge_detection(group: dict, det: dict) -> bool:
    gap = float(det["start"]) - float(group.get("end", 0.0))
    if gap > OCR_INTERVAL_SECONDS * 2.2:
        return False
    merged_duration = float(det["end"]) - float(group.get("start", det["start"]))
    if merged_duration > OCR_MAX_MERGED_SECONDS:
        return False
    if not similar_text(det["text"], group["text"]):
        return False
    if center_jumped(group.get("bbox", []), det.get("bbox", [])):
        return False
    return True


def merge_detections(detections: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for det in detections:
        matched = None
        for group in reversed(groups[-20:]):
            if can_merge_detection(group, det):
                matched = group
                break
        if matched:
            matched["end"] = det["end"]
            matched["confidence"] = max(float(matched.get("confidence", 0.0)), float(det.get("confidence", 0.0)))
            if len(det.get("text", "")) > len(matched.get("text", "")):
                matched["text"] = det["text"]
                matched["bbox"] = det.get("bbox", matched.get("bbox", []))
        else:
            item = dict(det)
            item["id"] = len(groups) + 1
            groups.append(item)
    return groups


def scan_video(base_dir: Path, video: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    duration, width, height = video_info(video)
    times = sample_times(duration)
    reader = make_reader()
    detections: list[dict] = []
    with ocr_temporary_directory(out_file) as tmp:
        tmp_dir = Path(tmp)
        frames_dir = tmp_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        total = len(times)
        frames: list[tuple[float, Path]] = []
        for index, timestamp in enumerate(times, start=1):
            write_status(base_dir, video.name, index, total)
            image = frames_dir / f"frame_{index:06d}.jpg"
            if not extract_frame(video, timestamp, image):
                continue
            frames.append((timestamp, image))
        image_results = reader.read_many([image for _, image in frames]) if hasattr(reader, "read_many") else {}
        for timestamp, image in frames:
            items = image_results.get(image) if image_results else reader.read(image)
            for item in items:
                detections.append(
                    {
                        "start": max(0.0, timestamp - OCR_INTERVAL_SECONDS / 2),
                        "end": timestamp + OCR_INTERVAL_SECONDS / 2,
                        "text": item["text"],
                        "confidence": item.get("confidence", 0.0),
                        "bbox": item.get("bbox", []),
                    }
                )
    data = {
        "video": video.name,
        "stem": video.stem,
        "duration": duration,
        "width": width,
        "height": height,
        "interval": OCR_INTERVAL_SECONDS,
        "detections": detections,
        "items": merge_detections(detections),
    }
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
