"""PDF 문자 분할과 OCR 병합에서 글자 중복·누락이 생기지 않는지 확인한다."""

import types
import unittest
from unittest.mock import patch

import test_api_flow
from src.services import document_input
from src.tasks.ocr_tasks import merge_native_text


class PdfTextTests(unittest.TestCase):
    def extract(self, characters):
        """문자와 PDF 좌표를 공급해 렌더링 좌표로 변환된 토큰을 받는다."""
        textpage = types.SimpleNamespace(count_chars=lambda: len(characters), get_charbox=lambda index, loose=False: characters[index][2] if loose and len(characters[index]) > 2 else characters[index][1])
        posconv = types.SimpleNamespace(to_bitmap=lambda x, y: (round(x * 2), round((100 - y) * 2)))

        with patch.object(document_input.pdfium.raw, "FPDFText_GetUnicode", side_effect=lambda page, index: ord(characters[index][0])):
            return document_input.pdf_text_tokens(textpage, posconv)

    def test_generated_line_breaks_between_letters_do_not_split_a_word(self):
        characters = []

        for index, character in enumerate("Receipt"):
            characters.extend([(character, (10 + index * 6, 50, 15 + index * 6, 60)), ("\r", (0, 0, 0, 0)), ("\n", (0, 0, 0, 0))])

        tokens = self.extract(characters)

        self.assertEqual([token["value"] for token in tokens], ["Receipt"])
        self.assertEqual(tokens[0]["bbox"], {"x_min": 20, "y_min": 80, "x_max": 102, "y_max": 100})
        self.assertIsNone(tokens[0]["score"])

    def test_real_rows_and_distant_columns_remain_separate(self):
        tokens = self.extract([("A", (10, 70, 15, 80)), (" ", (0, 0, 0, 0)), ("B", (20, 70, 25, 80)), ("C", (100, 70, 105, 80)), ("\n", (0, 0, 0, 0)), ("D", (10, 40, 15, 50))])

        self.assertEqual([token["value"] for token in tokens], ["A B", "C", "D"])

    def test_small_punctuation_does_not_split_the_following_letter(self):
        tokens = self.extract([("A", (10, 50, 15, 60)), (".", (16, 49, 17, 50), (16, 50, 17, 60)), ("\n", (0, 0, 0, 0)), ("b", (19, 51, 24, 58))])

        self.assertEqual([token["value"] for token in tokens], ["A.b"])

    def test_generated_breaks_preserve_a_visible_word_gap(self):
        tokens = self.extract([("A", (10, 50, 15, 60)), ("\n", (0, 0, 0, 0)), ("B", (20, 50, 25, 60))])

        self.assertEqual([token["value"] for token in tokens], ["A B"])

    def test_unicode_characters_are_not_limited_to_two_byte_text(self):
        tokens = self.extract([("한", (10, 50, 19, 60)), ("글", (20, 50, 29, 60)), ("😀", (30, 50, 39, 60))])

        self.assertEqual(tokens[0]["value"], "한글😀")

    def test_combined_pdf_boxes_remove_ocr_duplicate_and_keep_image_text(self):
        native = [{"value": letter, "source": "pdf_text", "bbox": {"x_min": index * 10, "y_min": 0, "x_max": index * 10 + 9, "y_max": 10}} for index, letter in enumerate("Receipt")]
        image_text = {"value": "Icon text", "source": "ocr", "bbox": {"x_min": 100, "y_min": 0, "x_max": 150, "y_max": 10}}
        page = {"texts": [{"value": "Receipt", "source": "ocr", "bbox": {"x_min": 0, "y_min": 0, "x_max": 70, "y_max": 10}}, image_text]}

        merge_native_text(page, native)

        self.assertEqual([token["value"] for token in page["texts"]], [*"Receipt", "Icon text"])
        self.assertEqual(page["texts"][-1]["source"], "ocr")
        self.assertEqual(page["text_source"], "pdf_text+ocr")

    def test_overlapping_pdf_boxes_are_not_counted_twice(self):
        native = [{"value": "PDF", "bbox": {"x_min": 0, "y_min": 0, "x_max": 30, "y_max": 10}}] * 3
        page = {"texts": [{"value": "PDF and image value", "source": "ocr", "bbox": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 10}}]}

        merge_native_text(page, native)

        self.assertTrue(any(token["value"] == "PDF and image value" for token in page["texts"]))

    def test_padded_ocr_box_is_removed_only_when_contained_text_matches(self):
        native = [{"value": "BILLED TO", "source": "pdf_text", "bbox": {"x_min": 10, "y_min": 10, "x_max": 90, "y_max": 20}}]

        for value, retained in (("BILLED TO", False), ("BILLED TO 123", True)):
            with self.subTest(value=value):
                page = {"texts": [{"value": value, "source": "ocr", "bbox": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 30}}]}

                merge_native_text(page, native)

                self.assertEqual(any(token.get("source") == "ocr" for token in page["texts"]), retained)


if __name__ == "__main__":
    unittest.main()
