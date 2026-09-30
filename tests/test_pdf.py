import tempfile
import unittest
from pathlib import Path
import pymupdf
from tools.report_pdf import export_report_pdf


class PDFTests(unittest.TestCase):
    def test_korean_text_references_and_page_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.pdf'
            export_report_pdf('# SUMMARY\n\n한글 투자 평가 테스트\n\n# REFERENCE\n\n테스트 원문', path)
            with pymupdf.open(path) as pdf:
                self.assertLessEqual(len(pdf), 5)
                self.assertIn('한글', pdf[0].get_text())
                self.assertIn('REFERENCE', pdf[-1].get_text())
                self.assertNotIn('\ufffd', ''.join(p.get_text() for p in pdf))

    def test_overflow_is_explicit_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oversized.pdf'
            with self.assertRaisesRegex(ValueError, '5페이지'):
                export_report_pdf('# SUMMARY\n\n' + ('한글 문장입니다. ' * 12000) + '\n# REFERENCE\n\n원문', path)
            self.assertFalse(path.exists())
