import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import load_json, save_json
from step07_qwen_all import (
    dialogue_batch_payload,
    parse_dialogue_batch_response,
    translate_dialogue_file,
)


class BatchClient:
    def __init__(self, response_factory=None) -> None:
        self.payloads = []
        self.response_factory = response_factory or self._complete_all
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @staticmethod
    def _complete_all(payload):
        return {
            "items": [
                {"id": item["id"], "zh": f"译文{item['id']}"}
                for item in payload["items"]
            ]
        }

    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "Identify recurring person names" in prompt:
            content = json.dumps({"entries": []}, ensure_ascii=False)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
        if "payload.items" not in prompt:
            content = json.dumps({"zh": "歌词译文"}, ensure_ascii=False)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
        payload = json.loads(prompt.split("\n", 1)[1])
        self.payloads.append(payload)
        content = json.dumps(self.response_factory(payload), ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def segments(count: int) -> list[dict]:
    return [
        {
            "start": index * 2.0,
            "end": index * 2.0 + 1.5,
            "text": f"Dialogue line {index + 1}.",
            "kind": "dialogue",
            "is_music": False,
        }
        for index in range(count)
    ]


class Step07DialogueBatchTests(unittest.TestCase):
    def test_payload_has_four_active_and_four_read_only_items_per_side(self):
        payload = dialogue_batch_payload(segments(12), [4, 5, 6, 7], {}, 4)

        self.assertEqual([item["id"] for item in payload["read_only_context_before"]], [1, 2, 3, 4])
        self.assertEqual([item["id"] for item in payload["items"]], [5, 6, 7, 8])
        self.assertEqual([item["id"] for item in payload["read_only_context_after"]], [9, 10, 11, 12])

    def test_duplicate_and_missing_ids_are_retried_individually(self):
        content = json.dumps(
            {
                "items": [
                    {"id": 5, "zh": "五"},
                    {"id": 5, "zh": "重复五"},
                    {"id": 6, "zh": "六"},
                    {"id": 99, "zh": "越界"},
                ]
            },
            ensure_ascii=False,
        )

        results, fallback_ids, issues = parse_dialogue_batch_response(content, [5, 6, 7, 8])

        self.assertEqual(results, {6: {"zh": "六"}})
        self.assertEqual(fallback_ids, [5, 7, 8])
        self.assertTrue(any("duplicate" in issue for issue in issues))
        self.assertTrue(any("unexpected" in issue for issue in issues))

    def test_dialogue_file_uses_batches_of_four(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "sample.json"
            output = base / "translated.json"
            save_json(source, {"segments": segments(6)})
            client = BatchClient()

            translate_dialogue_file(client, base, source, output, 4, 4)

            translated = load_json(output, {})["segments"]
            self.assertEqual([len(payload["items"]) for payload in client.payloads], [4, 2])
            self.assertEqual([item["zh"] for item in translated], [f"译文{index}" for index in range(1, 7)])

    def test_non_dialogue_breaks_the_active_batch(self):
        data = segments(6)
        data[2]["kind"] = "lyric"
        data[2]["is_music"] = True
        client = BatchClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "sample.json"
            output = base / "translated.json"
            save_json(source, {"segments": data})

            translate_dialogue_file(client, base, source, output, 4, 4)

        self.assertEqual([[item["id"] for item in payload["items"]] for payload in client.payloads], [[1, 2], [4, 5, 6]])

    def test_existing_chinese_is_preserved_and_only_gaps_are_translated(self):
        data = segments(4)
        data[1]["zh"] = "片源原有中文"
        data[1]["translation_origin"] = "embedded_chinese"
        client = BatchClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "sample.json"
            output = base / "translated.json"
            save_json(source, {"segments": data})

            translate_dialogue_file(client, base, source, output, 4, 4)

            translated = load_json(output, {})["segments"]
            self.assertEqual(translated[1]["zh"], "片源原有中文")
            self.assertEqual(translated[1]["translation_origin"], "embedded_chinese")
            self.assertEqual(translated[0]["translation_origin"], "qwen")
            self.assertEqual(translated[2]["translation_origin"], "qwen")
            self.assertEqual(
                [[item["id"] for item in payload["items"]] for payload in client.payloads],
                [[1], [3, 4]],
            )


if __name__ == "__main__":
    unittest.main()
