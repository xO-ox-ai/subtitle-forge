import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import ass_polish_helpers as polish
import step07_qwen_all as step07
import terminology_consistency as terminology


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

    def test_binding_terminology_replaces_known_name_variants(self):
        entries = [
            {
                "source_en": "Rocket",
                "preferred_zh": "火箭",
                "zh_aliases": ["阿炮", "罗克特", "罗基特"],
                "case_sensitive": True,
            },
            {
                "source_en": "Goose",
                "preferred_zh": "古斯",
                "zh_aliases": ["阿鹅"],
                "case_sensitive": True,
            },
        ]

        result, missing = terminology.enforce_terminology(
            "阿炮跟阿鹅说话。",
            "Rocket spoke to Goose.",
            entries,
        )

        self.assertEqual(result, "火箭跟古斯说话。")
        self.assertEqual(missing, [])

    def test_binding_terminology_falls_back_when_polisher_drops_a_name(self):
        entries = [{"source_en": "Rocket", "preferred_zh": "火箭", "case_sensitive": True}]

        result, missing = terminology.enforce_terminology(
            "他是摄影师。",
            "Rocket is a photographer.",
            entries,
            fallback_text="火箭是摄影师。",
        )

        self.assertEqual(result, "火箭是摄影师。")
        self.assertEqual(missing, [])

    def test_binding_terminology_allows_a_naturally_omitted_vocative(self):
        entries = [{"source_en": "Michael", "preferred_zh": "迈克尔", "case_sensitive": True}]

        result, missing = terminology.enforce_terminology(
            "你看看，粉丝都很爱你。",
            "Look around, Michael, your fans love you.",
            entries,
            fallback_text="你看看，粉丝都很爱你。",
        )

        self.assertEqual(result, "你看看，粉丝都很爱你。")
        self.assertEqual(missing, [])

    def test_binding_terminology_allows_nested_name_with_different_rendering(self):
        entries = [
            {"source_en": "Clade", "preferred_zh": "克莱德", "case_sensitive": True},
            {"source_en": "Baby Clade", "preferred_zh": "索奇宝宝", "case_sensitive": True},
        ]

        result, missing = terminology.enforce_terminology(
            "索奇宝宝说得对。",
            "Baby Clade is right.",
            entries,
        )

        self.assertEqual(result, "索奇宝宝说得对。")
        self.assertEqual(missing, [])

    def test_chinese_alias_does_not_match_a_different_english_name(self):
        entries = [
            {
                "source_en": "Greg",
                "preferred_zh": "格雷格",
                "zh_aliases": ["格雷格斯"],
                "case_sensitive": True,
            },
            {"source_en": "Gregers", "preferred_zh": "格雷格斯", "case_sensitive": True},
        ]

        result, missing = terminology.enforce_terminology(
            "格雷格斯来了。",
            "Gregers is here.",
            entries,
        )

        self.assertEqual(result, "格雷格斯来了。")
        self.assertEqual(missing, [])

    def test_binding_terminology_repairs_alias_only_ocr(self):
        entries = [
            {
                "source_en": "Rocket",
                "preferred_zh": "阿炮",
                "zh_aliases": ["火箭"],
                "scope": "all",
            }
        ]

        result, missing = terminology.enforce_terminology(
            "火箭",
            "",
            entries,
            segment_kind="ocr",
        )

        self.assertEqual(result, "阿炮")
        self.assertEqual(missing, [])

    def test_project_terminology_has_priority_over_auto_names(self):
        merged = terminology.merge_terminology_entries(
            [{"source_en": "Rocket", "preferred_zh": "阿炮"}],
            [{"source_en": "Rocket", "preferred_zh": "火箭"}],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["preferred_zh"], "阿炮")

    def test_recurring_name_candidates_are_file_wide(self):
        candidates = terminology.extract_recurring_name_candidates(
            [
                {"en": "Rocket, meet Goose.", "zh": "火箭，见见古斯。"},
                {"en": "Hey, Rocket.", "zh": "嘿，罗克特。"},
                {"en": "Goose is here.", "zh": "阿鹅来了。"},
                {"en": "You should leave.", "zh": "你该走了。"},
            ]
        )

        names = {item["source_en"] for item in candidates}
        self.assertIn("Rocket", names)
        self.assertIn("Goose", names)
        self.assertNotIn("You", names)

    def test_recurring_name_examples_sample_late_occurrences(self):
        candidates = terminology.extract_recurring_name_candidates(
            [
                {"en": "Rocket.", "zh": f"第{index}次叫阿炮。"}
                for index in range(10)
            ],
            max_examples=3,
        )

        self.assertEqual(
            [item["zh"] for item in candidates[0]["examples"]],
            ["第0次叫阿炮。", "第4次叫阿炮。", "第9次叫阿炮。"],
        )

    def test_custom_bilingual_style_is_detected_for_name_audit(self):
        lines = [
            "Dialogue: 0,0:00:01.00,0:00:02.00,Chs,,0,0,0,,"
            "迈克尔来了\\N{\\rEng}Michael is here.\n"
        ]

        targets = polish.extract_all_bilingual_name_targets(lines)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].style, "Chs")
        self.assertEqual(targets[0].zh, "迈克尔来了")
        self.assertEqual(targets[0].en, "Michael is here.")

    def test_step09_can_build_file_name_terminology_for_existing_ass(self):
        client = CaptureClient(
            '{"entries":[{"source_en":"Rocket","preferred_zh":"阿炮",'
            '"zh_aliases":["火箭","罗克特"]}]}'
        )
        entries = polish.request_file_name_terminology(
            client,
            "test-model",
            "Sample.Movie",
            [
                {
                    "source_en": "Rocket",
                    "occurrences": 3,
                    "examples": [
                        {"en": "Rocket is here.", "zh": "火箭来了。"},
                        {"en": "Hi, Rocket.", "zh": "你好，罗克特。"},
                    ],
                }
            ],
            0.0,
        )

        self.assertEqual(entries[0]["source_en"], "Rocket")
        self.assertEqual(entries[0]["preferred_zh"], "阿炮")
        self.assertEqual(entries[0]["zh_aliases"], ["火箭", "罗克特"])

    def test_step07_enforces_binding_name_after_model_translation(self):
        segment = {"is_music": False}
        zh = step07.apply_translation_result(
            None,
            segment,
            "My name is Rocket.",
            "dialogue",
            {"zh": "我叫阿炮。"},
            [
                {
                    "source": "Rocket",
                    "preferred_zh": "火箭",
                    "zh_aliases": ["阿炮"],
                    "case_sensitive": True,
                }
            ],
        )

        self.assertEqual(zh, "我叫火箭。")
        self.assertEqual(segment["zh"], "我叫火箭。")

    def test_ass_finalizer_deterministically_enforces_project_terminology(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            work = base / "temp"
            ass_file = base / "Sample.Movie.ass"
            ass_file.write_text(
                "Dialogue: 0,0:00:01.00,0:00:02.00,BILINGUAL,QWEN_ZH,0,0,0,,"
                "我叫火箭。\\N{\\fs34}My name is Rocket.\n",
                encoding="utf-8-sig",
            )
            (base / "subtitle_terminology.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "source_en": "Rocket",
                                "preferred_zh": "阿炮",
                                "zh_aliases": ["火箭"],
                                "series": "Sample Movie",
                                "scope": "all",
                                "case_sensitive": True,
                                "reviewed": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            stats = polish.enforce_ass_file_terminology(ass_file, base, work)

            self.assertEqual(stats["changed"], 1)
            self.assertIn("我叫阿炮。", ass_file.read_text(encoding="utf-8-sig"))

    def test_ass_finalizer_preserves_custom_style_tags_and_spacing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            work = base / "temp"
            ass_file = base / "Sample.Movie.ass"
            original = (
                "Dialogue: 0,0:00:01.00,0:00:02.00,Chs,,0,0,0,,"
                "-谢尔顿登  -来吧\\N{\\rEng}Sheldon?\n"
            )
            ass_file.write_text(original, encoding="utf-8-sig")
            (base / "subtitle_terminology.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "source_en": "Sheldon",
                                "preferred_zh": "谢尔登",
                                "zh_aliases": ["谢尔顿登"],
                                "series": "Sample Movie",
                                "scope": "all",
                                "case_sensitive": True,
                                "reviewed": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            stats = polish.enforce_ass_file_terminology(ass_file, base, work)
            rendered = ass_file.read_text(encoding="utf-8-sig")

            self.assertEqual(stats["changed"], 1)
            self.assertIn("-谢尔登  -来吧\\N{\\rEng}Sheldon?", rendered)
            self.assertNotIn(r"{\fs34}", rendered)


if __name__ == "__main__":
    unittest.main()
