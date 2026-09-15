import unittest

from step00_extract_embedded_subs import choose_streams


class StreamSelectionTests(unittest.TestCase):
    def test_text_track_before_bitmap_duplicate(self):
        streams = [
            {"index": 5, "codec_name": "dvd_subtitle", "tags": {"title": "English (U.S. Manga Corps)"}},
            {"index": 6, "codec_name": "subrip", "tags": {"title": "English (U.S. Manga Corps) - Unstyled"}},
            {"index": 7, "codec_name": "subrip", "tags": {"title": "English (fansub)"}},
        ]
        self.assertEqual(choose_streams(streams, prefer_sdh=True)[0]["index"], 6)

    def test_sdh_preference_survives_text_preference(self):
        streams = [
            {"index": 1, "codec_name": "dvd_subtitle", "tags": {"title": "English SDH"}},
            {"index": 2, "codec_name": "subrip", "tags": {"title": "English"}},
        ]
        self.assertEqual(choose_streams(streams, prefer_sdh=True)[0]["index"], 1)


if __name__ == "__main__":
    unittest.main()
