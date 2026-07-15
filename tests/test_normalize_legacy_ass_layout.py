import unittest

from normalize_legacy_ass_layout import normalize_legacy_layout


HEADER = [
    "[Script Info]",
    "ScriptType: v4.00+",
    "[V4+ Styles]",
    "Style: Default,Old Font,20",
    "[Events]",
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
]


class NormalizeLegacyAssLayoutTests(unittest.TestCase):
    def test_merges_chinese_then_english_into_current_bilingual_event(self):
        lines = HEADER + [
            "Dialogue: 0,0:00:01.00,0:00:03.00,Chinese,,0,0,0,,打倒他！",
            "Dialogue: 0,0:00:01.00,0:00:03.00,English,,0,0,0,,Bring him down!",
        ]

        result, stats = normalize_legacy_layout(lines)

        dialogues = [line for line in result if line.startswith("Dialogue:")]
        self.assertEqual(stats["pairs_merged"], 1)
        self.assertEqual(len(dialogues), 1)
        self.assertIn("BILINGUAL", dialogues[0])
        self.assertTrue(dialogues[0].endswith(r"打倒他！\N{\fs34}Bring him down!"))
        self.assertIn("PlayResX: 1920", result)
        self.assertTrue(any(line.startswith("Style: BILINGUAL,") for line in result))

    def test_pairing_matches_polisher_last_event_rule_and_keeps_extra_fragment(self):
        lines = HEADER + [
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Tell them what happened",
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,眼看着",
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,告诉他们发生了什么",
        ]

        result, stats = normalize_legacy_layout(lines)

        dialogues = [line for line in result if line.startswith("Dialogue:")]
        self.assertEqual(stats["pairs_merged"], 1)
        self.assertEqual(stats["legacy_singles"], 1)
        self.assertEqual(len(dialogues), 2)
        self.assertTrue(any(line.endswith("眼看着") for line in dialogues))
        self.assertTrue(
            any(line.endswith(r"告诉他们发生了什么\N{\fs34}Tell them what happened") for line in dialogues)
        )

    def test_unpaired_english_is_retained_with_current_english_size(self):
        lines = HEADER + [
            "Dialogue: 0,0:00:01.00,0:00:03.00,English,,0,0,0,,English only",
        ]

        result, stats = normalize_legacy_layout(lines)

        dialogues = [line for line in result if line.startswith("Dialogue:")]
        self.assertEqual(stats["pairs_merged"], 0)
        self.assertEqual(len(dialogues), 1)
        self.assertTrue(dialogues[0].endswith(r"{\fs34}English only"))

    def test_music_pair_uses_current_music_style(self):
        lines = HEADER + [
            "Dialogue: 0,0:00:01.00,0:00:03.00,English,,0,0,0,,♪ Sing ♪",
            "Dialogue: 0,0:00:01.00,0:00:03.00,Chinese,,0,0,0,,♪ 歌唱 ♪",
        ]

        result, _stats = normalize_legacy_layout(lines)

        dialogue = next(line for line in result if line.startswith("Dialogue:"))
        self.assertIn(",BILINGUAL_MUSIC,", dialogue)

    def test_zh_en_alias_pair_normalizes_to_current_layout(self):
        lines = HEADER + [
            "Dialogue: 0,0:00:01.00,0:00:03.00,ZH,,0,0,0,,把球传过来！",
            "Dialogue: 0,0:00:01.00,0:00:03.00,EN,,0,0,0,,Pass me the ball!",
        ]

        result, stats = normalize_legacy_layout(lines)

        dialogues = [line for line in result if line.startswith("Dialogue:")]
        self.assertEqual(stats["pairs_merged"], 1)
        self.assertEqual(len(dialogues), 1)
        self.assertIn(",BILINGUAL,", dialogues[0])
        self.assertTrue(dialogues[0].endswith(r"把球传过来！\N{\fs34}Pass me the ball!"))


if __name__ == "__main__":
    unittest.main()
