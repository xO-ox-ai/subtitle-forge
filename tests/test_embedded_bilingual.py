import tempfile
import unittest
from collections import Counter
from pathlib import Path

from common import iter_videos
from embedded_bitmap_ocr import split_joined_ocr_words
from step00_extract_embedded_bilingual import align_bilingual, choose_chinese, english_segments
from step00_extract_embedded_subs import collect_sdh_labels, make_segment, normalize_compare
from step08_ass_render import coalesce_text_chunks, display_parts, song_translation_key


def cue(start, end, text, **extra):
    return {"start": start, "end": end, "text": text, "en_wrap": text, **extra}


class EmbeddedBilingualTests(unittest.TestCase):
    def test_bitmap_ocr_splits_likely_joined_words_only(self):
        dictionary = {"of", "never", "their", "best", "firsthand", "with", "out"}
        text = split_joined_ocr_words(
            "ofnever theirbest firsthand without",
            dictionary,
        )

        self.assertEqual(text, "of never their best firsthand without")

    def test_video_discovery_is_recursive(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "Season 01"
            nested.mkdir()
            video = nested / "Episode.mkv"
            video.touch()

            self.assertEqual(iter_videos(Path(directory)), [video])

    def test_uses_english_timing_and_merges_split_english(self):
        english = [
            cue(1.000, 2.000, "Hello.", is_music=False, is_chant=False),
            cue(2.050, 3.000, "How are you?", is_music=False, is_chant=False),
        ]
        chinese = [cue(1.030, 3.040, "你好，你怎么样？")]

        segments, counts = align_bilingual(english, chinese)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 1.0)
        self.assertEqual(segments[0]["end"], 3.0)
        self.assertEqual(segments[0]["text"], "Hello. How are you?")
        self.assertEqual(segments[0]["zh"], "你好，你怎么样？")
        self.assertEqual(segments[0]["translation_origin"], "embedded_chinese")
        self.assertEqual(counts["matched_english"], 2)

    def test_discards_unmatched_sdh_effect_but_keeps_music(self):
        english = [
            cue(1.0, 2.0, "Dialogue", is_music=False, is_chant=False),
            cue(5.0, 7.0, "♪ Song ♪", is_music=True, is_chant=False),
        ]
        segments, counts = align_bilingual(english, [])

        self.assertEqual([segment["text"] for segment in segments], ["♪ Song ♪"])
        self.assertEqual(segments[0]["translation_origin"], "missing")
        self.assertEqual(counts["unmatched_english_removed"], 1)

    def test_prefers_simplified_chinese(self):
        streams = [
            {"index": 1, "tags": {"language": "chi", "title": "Traditional"}},
            {"index": 2, "tags": {"language": "chi", "title": "Simplified"}},
        ]
        self.assertEqual(choose_chinese(streams)["index"], 2)

    def test_mixed_dialogue_and_background_song_keeps_dialogue_only(self):
        source = cue(
            10.0,
            12.0,
            "I'll be fine. - ♪ Background song ♪",
            raw="I'll be fine.\n- ♪ Background song ♪",
            labels=[],
            pure_label=False,
            italic=False,
            has_music=True,
        )

        segments = english_segments([source])

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "I'll be fine.")
        self.assertFalse(segments[0]["is_music"])

    def test_song_translation_key_removes_music_marks_and_sdh_dash(self):
        self.assertEqual(song_translation_key("- ♪ Would you help me? ♪"), "Would you help me?")
    def test_adjacent_music_marker_does_not_mark_previous_dialogue(self):
        dialogue = cue(10.0, 12.0, "Regular dialogue")
        marker = cue(12.08, 14.0, "♪", labels=["Singing"], pure_label=False)

        self.assertEqual(collect_sdh_labels(dialogue, [marker]), [])

    def test_repeated_plain_english_is_not_a_chant(self):
        text = "Hold on. Hold on. We'll work that out later."
        source = cue(1.0, 3.0, text, raw=text, italic=False, has_music=False)

        segment = make_segment(source, [], Counter({normalize_compare(text): 1}), "embedded", 1)

        self.assertEqual(segment["kind"], "dialogue")
        self.assertFalse(segment["is_chant"])

    def test_orphan_music_marker_does_not_mark_spoken_line(self):
        text = "- ♪ - You just relax back, okay?"
        source = cue(1.0, 3.0, text, raw=text, italic=True, has_music=True)

        segment = make_segment(source, [], Counter({normalize_compare(text): 1}), "embedded", 1)

        self.assertEqual(segment["text"], "- You just relax back, okay?")
        self.assertEqual(segment["kind"], "dialogue")
        self.assertFalse(segment["is_music"])

    def test_short_embedded_cue_without_word_times_is_not_micro_split(self):
        segment = {
            "text": "One, two, three! One, two, three! One, two, one, two, one, two!",
            "zh": "一、二、三，一、二、三，一、二、一、二、一、二！",
            "is_music": False,
            "is_chant": True,
        }

        parts = display_parts(segment, 10.0, 14.2, segment["zh"], segment["text"])

        self.assertEqual(len(parts), 1)
        self.assertIn("一、二、三", parts[0]["zh"])

    def test_long_embedded_bilingual_block_is_not_split_by_character_ratio(self):
        segment = {
            "text": (
                "Of course they were entertained. It was like true modern society. "
                "Drama, suspense! Politics are a real goldmine. "
                "I'm relieved that 8th Floor doesn't seem upset."
            ),
            "zh": "怎么不算呢？我从昨天开始就忙着四处拉票。8楼看上去没有不高兴。",
            "translation_origin": "embedded_chinese",
            "is_music": False,
            "is_chant": False,
        }

        parts = display_parts(segment, 10.0, 21.5, segment["zh"], segment["text"])

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["zh"], segment["zh"])
        self.assertEqual(parts[0]["en"], segment["text"])

    def test_untimed_long_dialogue_limits_display_chunks(self):
        chunks = [f"Sentence {index}." for index in range(12)]

        merged = coalesce_text_chunks(chunks, 4)

        self.assertEqual(len(merged), 4)
        self.assertEqual(" ".join(merged), " ".join(chunks))


if __name__ == "__main__":
    unittest.main()
