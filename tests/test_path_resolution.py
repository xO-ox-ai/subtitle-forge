import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_common as paths


class PathResolutionTests(unittest.TestCase):
    def test_relative_path_is_resolved_from_project_root(self):
        self.assertEqual(
            paths.resolve_project_path(Path("models") / "sample.bin"),
            (paths.PROJECT_ROOT / "models" / "sample.bin").resolve(),
        )

    def test_absolute_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            absolute = Path(temp_dir).resolve()
            self.assertEqual(paths.resolve_project_path(absolute), absolute)

    def test_relative_environment_path_is_project_relative(self):
        with patch.dict(os.environ, {"SUB_TEST_PATH": ".cache/test"}):
            self.assertEqual(
                paths.env_path("SUB_TEST_PATH", "unused"),
                (paths.PROJECT_ROOT / ".cache" / "test").resolve(),
            )

    def test_executable_prefers_path_before_fallback(self):
        path_hit = str(paths.PROJECT_ROOT / "path-bin" / "ffmpeg.exe")
        fallback = paths.PROJECT_ROOT / "tools" / "ffmpeg" / "bin"
        with patch("pipeline_common.shutil.which", return_value=path_hit):
            self.assertEqual(paths.exe_path("ffmpeg.exe", fallback_dirs=(fallback,)), path_hit)

    def test_relative_executable_override_is_project_relative(self):
        with patch.dict(os.environ, {"SUB_TEST_EXE": "tools/custom/tool.exe"}):
            with patch("pipeline_common.shutil.which", return_value=None):
                self.assertEqual(
                    paths.exe_path("tool.exe", "SUB_TEST_EXE"),
                    str((paths.PROJECT_ROOT / "tools" / "custom" / "tool.exe").resolve()),
                )

    def test_bare_executable_override_remains_path_command(self):
        with patch("pipeline_common.shutil.which", return_value=None):
            self.assertEqual(paths.resolve_executable_path("custom-python"), Path("custom-python"))

    def test_specialized_environment_falls_back_next_to_path_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_root = root / "project"
            runtime_root = root / "runtime"
            adjacent_scripts = runtime_root / "paddleocr" / "Scripts"
            project_root.mkdir()
            adjacent_scripts.mkdir(parents=True)
            with patch.object(paths, "PROJECT_ROOT", project_root):
                with patch.object(paths, "_CURRENT_ROOT", runtime_root):
                    with patch.dict(os.environ, {}, clear=False):
                        os.environ.pop("SUB_TEST_SCRIPTS", None)
                        self.assertEqual(
                            paths.specialized_scripts_dir("SUB_TEST_SCRIPTS", "paddleocr"),
                            adjacent_scripts.resolve(),
                        )


if __name__ == "__main__":
    unittest.main()
