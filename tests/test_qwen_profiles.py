import os
import unittest
from unittest.mock import patch

import qwen_common


class QwenProfileTests(unittest.TestCase):
    def tearDown(self):
        qwen_common.configure_qwen_profile(qwen_common.DEFAULT_QWEN_PROFILE)

    def test_80b_profile_uses_safe_partial_gpu_defaults(self):
        with patch.dict(os.environ, {"QWEN_GPU_LAYERS": "all"}, clear=False):
            profile = qwen_common.configure_qwen_profile("80b")
            command = qwen_common.build_server_cmd()

        self.assertEqual(profile["profile"], "80b")
        self.assertTrue(profile["gguf"].endswith("Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"))
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "22")
        self.assertEqual(command[command.index("--threads") + 1], "16")
        self.assertEqual(command[command.index("--cache-ram") + 1], "2048")

    def test_32b_profile_remains_available_as_explicit_fallback(self):
        with patch.dict(
            os.environ,
            {
                "QWEN_32B_GPU_LAYERS": "all",
                "QWEN_32B_GGUF": "",
            },
            clear=False,
        ):
            profile = qwen_common.configure_qwen_profile("32b")
            command = qwen_common.build_server_cmd()

        self.assertEqual(profile["model"], "qwen3-32b")
        self.assertTrue(profile["gguf"].endswith("Qwen3-32B-Q4_K_M.gguf"))
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "all")

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            qwen_common.configure_qwen_profile("unknown")


if __name__ == "__main__":
    unittest.main()
