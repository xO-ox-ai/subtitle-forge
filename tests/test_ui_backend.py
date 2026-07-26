import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import selected_videos
from forge_ui.config import AppSettings, load_settings, save_settings
from forge_ui.environment import HardwareInfo, recommend_qwen
from forge_ui.planner import VideoTask, build_pipeline_commands


class UIBackendTests(unittest.TestCase):
    def test_settings_secret_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            settings = AppSettings(hf_token="hf_example", polish_api_key="sk_example")
            save_settings(settings, path)
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("hf_example", raw)
            self.assertNotIn("sk_example", raw)
            loaded = load_settings(path)
            self.assertEqual("hf_example", loaded.hf_token)
            self.assertEqual("sk_example", loaded.polish_api_key)

    def test_cloud_polish_requires_key_for_openai(self):
        settings = AppSettings(polish_provider="openai", polish_api_key="")
        self.assertFalse(settings.cloud_polish_available)
        settings.polish_api_key = "sk-example"
        self.assertTrue(settings.cloud_polish_available)

    def test_portable_runtime_is_injected_into_child_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "runtime" / "demucs" / "python.exe"
            scripts = python.parent / "Scripts"
            scripts.mkdir(parents=True)
            python.touch()
            with patch.dict(os.environ, {"SUBTITLE_FORGE_HOME": temp}):
                env = AppSettings().child_environment()
            self.assertEqual(str(python), env["SUB_PYTHON_DEMUCS_EXE"])
            self.assertEqual(str(scripts), env["SUB_DEMUCS_SCRIPTS"])

    def test_hardware_recommendation(self):
        self.assertEqual("80b", recommend_qwen(HardwareInfo(64, "RTX", 24))[0])
        self.assertEqual("32b", recommend_qwen(HardwareInfo(32, "RTX", 12))[0])

    def test_task_flags_become_pipeline_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            video = base / "Season 1" / "Episode 01.mkv"
            video.parent.mkdir()
            video.touch()
            settings = AppSettings(qwen_profile="32b", polish_api_key="")
            task = VideoTask(
                path=video,
                group="no_subtitles",
                use_ocr=False,
                use_polish=True,
                cleanup=True,
            )
            command = build_pipeline_commands(base, [task], settings)[0]
            self.assertIn("--skip-ocr", command.argv)
            self.assertIn("--no-polish", command.argv)
            self.assertIn("--cleanup", command.argv)
            self.assertIn(str(Path("Season 1") / "Episode 01.mkv"), command.argv)

    def test_relative_path_selects_only_one_duplicate_stem(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            first = base / "A" / "Episode.mkv"
            second = base / "B" / "Episode.mkv"
            first.parent.mkdir()
            second.parent.mkdir()
            first.touch()
            second.touch()
            selected = selected_videos(base, target_stems=[str(Path("B") / "Episode.mkv")])
            self.assertEqual([second], selected)


if __name__ == "__main__":
    unittest.main()
