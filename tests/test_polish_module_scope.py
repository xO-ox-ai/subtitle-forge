import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ass_polish_helpers as polish


class PolishModuleScopeTests(unittest.TestCase):
    def test_polish_initializes_terms_and_reaches_request_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            ass = base / 'episode.ass'
            ass.write_text('Dialogue: 0,0:00:01.00,0:00:03.00,BILINGUAL,QWEN_ZH,0,0,0,,你好\\NHello\n', encoding='utf-8')
            with patch.object(polish, 'build_polish_client', return_value=object()), \
                 patch.object(polish, 'load_or_create_file_name_terminology', return_value=[]), \
                 patch.object(polish, 'request_polish_batch', return_value=None) as request:
                result = polish.polish_ass_files(
                    [ass], base, base / 'temp', provider='codex-cli', model='test',
                    base_url='', api_key='', batch_size=4, timeout=10,
                    temperature=0.2, style_names={'BILINGUAL'},
                )
            request.assert_called_once()
            self.assertEqual(result[ass]['requested'], 1)
            self.assertEqual(result[ass]['failed'], 1)


if __name__ == '__main__':
    unittest.main()
