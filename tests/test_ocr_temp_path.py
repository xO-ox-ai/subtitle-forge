import tempfile
import unittest
from pathlib import Path

import ocr_extract_core


class OcrTempPathTests(unittest.TestCase):
    def test_ocr_scratch_directory_is_inside_work_dir_and_removed(self):
        with tempfile.TemporaryDirectory() as outer:
            work_dir = Path(outer) / "temp"
            out_file = work_dir / "ocr_raw" / "episode.json"

            with ocr_extract_core.ocr_temporary_directory(out_file) as scratch:
                scratch_path = Path(scratch)
                self.assertEqual(scratch_path.parent, work_dir)
                self.assertTrue(scratch_path.exists())

            self.assertFalse(scratch_path.exists())

    def test_ocr_scratch_directory_is_removed_after_exception(self):
        with tempfile.TemporaryDirectory() as outer:
            work_dir = Path(outer) / "temp"
            out_file = work_dir / "ocr_raw" / "episode.json"
            scratch_path = None

            with self.assertRaises(RuntimeError):
                with ocr_extract_core.ocr_temporary_directory(out_file) as scratch:
                    scratch_path = Path(scratch)
                    raise RuntimeError("test failure")

            self.assertIsNotNone(scratch_path)
            self.assertFalse(scratch_path.exists())


if __name__ == "__main__":
    unittest.main()
