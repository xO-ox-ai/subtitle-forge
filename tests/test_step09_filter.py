import tempfile
import unittest
from pathlib import Path

import step09_overlay_filter as step09


class Step09FilterTests(unittest.TestCase):
    def test_separator_only_bilingual_is_removed(self):
        separator = r"Dialogue: 0,0:00:01.00,0:00:02.00,BILINGUAL,,0,0,0,,——\N{\fs34}- -"
        dialogue = r"Dialogue: 0,0:00:02.00,0:00:03.00,BILINGUAL,,0,0,0,,别走\N{\fs34}- Don't go."
        with tempfile.TemporaryDirectory() as temp_dir:
            ass_file = Path(temp_dir) / "sample.ass"
            ass_file.write_text("[Events]\n" + separator + "\n" + dialogue + "\n", encoding="utf-8-sig")

            _ocr_total, removed, _credits = step09.filter_ass_file(ass_file)

            output = ass_file.read_text(encoding="utf-8-sig")
            self.assertEqual(removed, 1)
            self.assertNotIn(separator, output)
            self.assertIn(dialogue, output)

    def test_real_dash_prefixed_dialogue_is_not_separator(self):
        line = r"Dialogue: 0,0:00:01.00,0:00:02.00,BILINGUAL,,0,0,0,,- 我不去\N{\fs34}- I'm not going."
        self.assertFalse(step09.is_separator_only_bilingual(line))

    def test_ellipsis_and_sdh_dashes_are_separator(self):
        line = r"Dialogue: 0,0:00:01.00,0:00:02.00,BILINGUAL,,0,0,0,,……\N{\fs34}- -"
        self.assertTrue(step09.is_separator_only_bilingual(line))


if __name__ == "__main__":
    unittest.main()
