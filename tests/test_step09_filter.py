import tempfile
import unittest
from pathlib import Path

import step09_overlay_filter as step09
from ass_polish_helpers import extract_polish_targets, parse_polish_styles


class Step09FilterTests(unittest.TestCase):
    def test_selected_ass_files_discovers_nested_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            nested_dir = base_dir / "Show" / "Season 03"
            nested_dir.mkdir(parents=True)
            nested_ass = nested_dir / "Show.S03E03.ass"
            nested_ass.write_text("[Events]\n", encoding="utf-8-sig")

            selected = step09.selected_ass_files(base_dir, "0", ["S03E03"])

            self.assertEqual(selected, [nested_ass])

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


    def test_polish_origin_filter_protects_embedded_chinese(self):
        lines = [
            r"Dialogue: 0,0:00:01.00,0:00:02.00,BILINGUAL,EMBEDDED_CHINESE,0,0,0,,原有中文\N{\fs34}Original",
            r"Dialogue: 0,0:00:02.00,0:00:03.00,BILINGUAL,QWEN_ZH,0,0,0,,本地译文\N{\fs34}Translated",
            r"Dialogue: 0,0:00:02.50,0:00:03.50,BILINGUAL,MANUAL_ZH,0,0,0,,人工译文\N{\fs34}Manual",
            r"Dialogue: 5,0:00:03.00,0:00:04.00,OCR_TRANSLATION,,0,0,0,,屏幕文字",
            r"Dialogue: 6,0:00:04.00,0:00:05.00,EXPLANATION_NOTE,,0,0,0,,文化注释",
        ]

        targets = extract_polish_targets(lines, parse_polish_styles("all"), {"qwen", "ocr"})

        self.assertEqual([target.origin for target in targets], ["qwen", "ocr"])


if __name__ == "__main__":
    unittest.main()
