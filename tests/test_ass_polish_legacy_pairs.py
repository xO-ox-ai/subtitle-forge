import unittest

from ass_polish_helpers import (
    DEFAULT_STYLES,
    extract_polish_targets,
    parse_polish_styles,
    replace_target_zh,
)


class LegacyPairPolishTests(unittest.TestCase):
    def test_default_style_list_includes_zh_aliases(self):
        styles = parse_polish_styles(DEFAULT_STYLES)

        self.assertIn("ZH", styles)
        self.assertIn("ZH_i", styles)

    def test_separate_english_and_chinese_events_become_one_target(self):
        lines = [
            "Dialogue: 0,0:00:01.00,0:00:03.00,English,,0,0,0,,Bring him down!",
            "Dialogue: 0,0:00:01.00,0:00:03.00,Chinese,,0,0,0,,把他拉下来!",
        ]

        targets = extract_polish_targets(lines, parse_polish_styles(""))

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].line_index, 1)
        self.assertEqual(targets[0].kind, "dialogue")
        self.assertEqual(targets[0].en, "Bring him down!")
        self.assertEqual(targets[0].zh, "把他拉下来!")

    def test_default_style_extra_fragment_prefers_last_chinese_event(self):
        lines = [
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Tell your grandchildren that you were there",
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,眼看着黑",
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,告诉你的子孙你在场",
        ]

        targets = extract_polish_targets(lines, parse_polish_styles(""))

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].line_index, 2)
        self.assertEqual(targets[0].zh, "告诉你的子孙你在场")

    def test_unpaired_legacy_event_is_not_sent_without_source(self):
        lines = ["Dialogue: 0,0:00:01.00,0:00:03.00,Chinese,,0,0,0,,只有中文"]

        self.assertEqual(extract_polish_targets(lines, parse_polish_styles("")), [])

    def test_replacement_changes_only_the_chinese_event_text(self):
        lines = [
            "Dialogue: 0,0:00:01.00,0:00:03.00,English,,0,0,0,,Bring him down!",
            "Dialogue: 0,0:00:01.00,0:00:03.00,Chinese,,0,0,0,,{\\i1}把他拉下来!",
        ]
        target = extract_polish_targets(lines, parse_polish_styles(""))[0]

        replaced = replace_target_zh(lines[1], target, "把他拽下来！")

        self.assertEqual(
            replaced,
            "Dialogue: 0,0:00:01.00,0:00:03.00,Chinese,,0,0,0,,{\\i1}把他拽下来！",
        )

    def test_zh_en_aliases_are_paired_and_music_is_classified(self):
        lines = [
            "Dialogue: 0,0:00:01.00,0:00:03.00,EN_i,,0,0,0,,♪ Sing it loud ♪",
            "Dialogue: 0,0:00:01.00,0:00:03.00,ZH_i,,0,0,0,,♪ 大声唱 ♪",
        ]

        targets = extract_polish_targets(lines, parse_polish_styles(""))

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].style, "ZH_i")
        self.assertEqual(targets[0].kind, "lyric")
        self.assertEqual(targets[0].en, "♪ Sing it loud ♪")


if __name__ == "__main__":
    unittest.main()
