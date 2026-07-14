import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import pipeline_common
import step10_quality_report


class RuntimePathTests(unittest.TestCase):
    def test_status_and_log_are_written_under_temp(self):
        with tempfile.TemporaryDirectory() as outer:
            base_dir = Path(outer)
            common.write_status(base_dir, "TEST", "sample.mkv", "running")

            self.assertTrue((base_dir / "temp" / "subtitle_status.txt").exists())
            self.assertTrue((base_dir / "temp" / "subtitle_run.log").exists())
            self.assertFalse((base_dir / "subtitle_status.txt").exists())
            self.assertFalse((base_dir / "subtitle_run.log").exists())

    def test_manifest_is_written_under_work_dir(self):
        with tempfile.TemporaryDirectory() as outer:
            base_dir = Path(outer)
            work_dir = base_dir / "temp"
            video = base_dir / "sample.mkv"

            manifest = step10_quality_report.write_manifest(base_dir, work_dir, [video])

            self.assertEqual(manifest, work_dir / "pipeline_manifest.json")
            self.assertTrue(manifest.exists())
            self.assertFalse((base_dir / "pipeline_manifest.json").exists())

    def test_generic_temp_environment_uses_project_temp(self):
        with tempfile.TemporaryDirectory() as outer:
            target = Path(outer) / "temp"
            previous_tempdir = tempfile.tempdir
            try:
                with patch.object(pipeline_common, "PROJECT_TEMP_DIR", target):
                    with patch.dict(os.environ, {}, clear=False):
                        pipeline_common.configure_environment()
                        self.assertEqual(os.environ["TEMP"], str(target))
                        self.assertEqual(os.environ["TMP"], str(target))
                        self.assertEqual(os.environ["TMPDIR"], str(target))
                        self.assertEqual(os.environ["PYTHONDONTWRITEBYTECODE"], "1")
                        self.assertEqual(tempfile.tempdir, str(target))
                        self.assertTrue(sys.dont_write_bytecode)
                        self.assertTrue(target.is_dir())
            finally:
                tempfile.tempdir = previous_tempdir


if __name__ == "__main__":
    unittest.main()
