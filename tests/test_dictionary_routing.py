import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import ass_polish_helpers as polish
import step07_qwen_all as step07


class CaptureClient:
    def __init__(self, content: str = '{"items":[]}') -> None:
        self.messages = []
        self.content = content

    def chat_json(self, **kwargs):
        self.messages = kwargs["messages"]
        return {"choices": [{"message": {"content": self.content}}]}


def target(en: str, zh: str = "旧译") -> polish.PolishTarget:
    return polish.PolishTarget(
        line_index=0,
        style="BILINGUAL",
        kind="dialogue",
        start="0:00:01.00",
        end="0:00:02.00",
        text_field="",
        zh=zh,
        en=en,
    )


class DictionaryRoutingTests(unittest.TestCase):
    def test_terminology_is_exact_and_series_scoped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "subtitle_terminology.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "source_en": "locator spell",
                                "aliases": ["locator spells"],
                                "preferred_zh": "定位咒",
                                "note": "不要照抄这条说明",
                                "series": "The Originals",
                                "scope": "dialogue|chant",
                                "reviewed": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            originals_state = SimpleNamespace(ass_file=Path("The.Originals.S04E01.ass"))
            nashville_state = SimpleNamespace(ass_file=Path("Nashville.S04E01.ass"))
            cue = target("Try a locator spell.")

            selected = polish.load_relevant_terminology(base, [(originals_state, cue)])
            self.assertEqual(
                selected,
                [
                    {
                        "source_en": "locator spell",
                        "preferred_zh": "定位咒",
                        "note": "不要照抄这条说明",
                    }
                ],
            )
            self.assertEqual(polish.load_relevant_terminology(base, [(nashville_state, cue)]), [])
            plural = target("Try every locator spells ritual.")
            self.assertEqual(
                polish.load_relevant_terminology(base, [(originals_state, plural)])[0]["preferred_zh"],
                "定位咒",
            )

    def test_case_sensitive_terminology_does_not_capture_ordinary_lowercase(self):
        entry = {
            "source_en": "the Hollow",
            "aliases": ["Hollow"],
            "preferred_zh": "空洞",
            "case_sensitive": True,
        }
        self.assertTrue(polish.terminology_entry_matches(entry, "The Hollow is coming."))
        self.assertTrue(polish.terminology_entry_matches(entry, "Hollow magic is spreading."))
        self.assertFalse(polish.terminology_entry_matches(entry, "a hollow promise"))

    def test_cultural_glossary_is_not_polish_context(self):
        client = CaptureClient()
        polish.request_polish_batch(
            client,
            "test-model",
            [("The.Originals.S04E01.ass", target("Try a locator spell."))],
            [{"source_en": "locator spell", "preferred_zh": "定位咒", "note": "说明"}],
            [{"term": "spell", "guidance": "按巫术语境处理"}],
            [{"phrase": "try", "guidance": "按上下文处理"}],
            0.0,
        )
        prompt = client.messages[-1]["content"]
        payload = json.loads(prompt.split("\n", 1)[1])
        self.assertNotIn("glossary", payload)
        self.assertEqual(payload["terminology"][0]["preferred_zh"], "定位咒")
        self.assertEqual(payload["mistranslation_hints"][0]["guidance"], "按巫术语境处理")
        self.assertEqual(payload["phrase_hints"][0]["guidance"], "按上下文处理")

    def test_terminology_note_echo_is_rejected(self):
        client = CaptureClient('{"items":[{"id":1,"zh":"这是一条使用说明"}]}')
        result = polish.request_polish_batch(
            client,
            "test-model",
            [("The.Originals.S04E01.ass", target("Try a locator spell."))],
            [{"source_en": "locator spell", "preferred_zh": "定位咒", "note": "这是一条使用说明"}],
            [],
            [],
            0.0,
        )
        self.assertEqual(result, {})

    def test_step07_does_not_load_cultural_glossary_as_translation_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "subtitle_glossary.json").write_text(
                json.dumps({"terms": [{"term": "locator spell", "zh": "文化注解正文"}]}),
                encoding="utf-8",
            )
            (base / "subtitle_terminology.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "source_en": "locator spell",
                                "preferred_zh": "定位咒",
                                "series": "The Originals",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            guidance = step07.load_step09_guidance(base, "The.Originals.S04E01")
            self.assertNotIn("glossary", guidance)
            self.assertEqual(guidance["terminology"][0]["preferred_zh"], "定位咒")

    def test_static_hint_review_ignores_stale_prompt_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "sample.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "old": {
                                "prompt_version": "ass-polish-older",
                                "kind": "dialogue",
                                "en": "old source",
                                "source_zh": "旧译",
                                "polished_zh": "旧改",
                            },
                            "current": {
                                "prompt_version": polish.PROMPT_VERSION,
                                "kind": "dialogue",
                                "en": "current source",
                                "source_zh": "原译",
                                "polished_zh": "新译",
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            samples = polish._collect_static_hint_samples([cache])
            self.assertEqual(len(samples["changed_items"]), 1)
            self.assertEqual(samples["changed_items"][0]["en"], "current source")


if __name__ == "__main__":
    unittest.main()
