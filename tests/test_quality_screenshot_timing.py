import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from step10_quality_report import capture_screenshot


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for the render integration test')
class ScreenshotTimingTests(unittest.TestCase):
    def test_seek_preserves_subtitle_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, ass, shot = root / 'black.mkv', root / 'text.ass', root / 'shot.png'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=black:s=320x180:r=10', '-t', '3', '-c:v', 'ffv1', str(video)], check=True)
            ass.write_text('[Script Info]\nScriptType: v4.00+\nPlayResX: 320\nPlayResY: 180\n'
                           '[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, Alignment\n'
                           'Style: Default,Arial,24,&H00FFFFFF,5\n'
                           '[Events]\nFormat: Layer, Start, End, Style, Text\n'
                           'Dialogue: 0,0:00:01.00,0:00:02.00,Default,VISIBLE\n', encoding='utf-8')
            self.assertTrue(capture_screenshot(video, ass, 1.5, shot))
            pixels = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(shot), '-f', 'rawvideo',
                                     '-pix_fmt', 'gray', '-'], check=True, stdout=subprocess.PIPE).stdout
            self.assertGreater(sum(pixels), 10000, 'Expected white subtitle pixels on the black frame')


if __name__ == '__main__':
    unittest.main()
