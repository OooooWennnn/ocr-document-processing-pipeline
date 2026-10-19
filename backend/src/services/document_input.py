"""Resolve storage paths and read PDF pages, text and pixel coordinates."""

import os
from pathlib import Path
from contextlib import closing
import pypdfium2 as pdfium
from PIL import Image

MAX_PAGES = 30
MAX_PAGE_PIXELS = 25_000_000


def resolve_source_path(source):
    """Resolve existing or legacy storage paths without changing files or database rows."""

    path = Path(source)

    if path.is_file():
        return path

    legacy = os.getenv("OCR_STORAGE_LEGACY_ROOT")
    storage = os.getenv("OCR_STORAGE_DIR")

    if legacy and storage:
        try:
            relative = path.relative_to(Path(legacy))
        except ValueError:
            return path

        root = Path(storage).resolve()
        mapped = (root / relative).resolve()

        if mapped.is_relative_to(root):
            return mapped

    return path


def validate_pdf(content):
    """Return the validated PDF page count; reject unreadable files, unsupported encryption and invalid pages."""
    try:
        with pdfium.PdfDocument(content) as pdf:
            count = len(pdf)

            if not 1 <= count <= MAX_PAGES:
                raise ValueError(f"PDFs must contain 1 to {MAX_PAGES} pages.")

            for index in range(count):
                with closing(pdf[index]) as page:
                    width, height = page.get_size()

                    if width <= 0 or height <= 0:
                        raise ValueError("Invalid PDF page size.")

            return count
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be read. Password-protected PDFs are not supported.") from exc


def preview_path(source, page_index):
    """Build the preview path for a zero-based PDF page."""
    return Path(source).with_suffix("") / f"page-{page_index}.png"


def pdf_text_tokens(textpage, posconv):
    """PDF 문자 순서와 공백을 유지해 줄 단위 토큰을 반환한다. 멀리 떨어진 필드는 분리한다."""
    tokens = []
    characters = []
    boxes = []
    line_break = False
    previous_font = None

    def finish_line():
        """모은 글자와 좌표를 토큰 하나로 저장하고 다음 줄을 준비한다."""
        value = "".join(characters).strip()

        if value and boxes:
            left, bottom = min(box[0] for box in boxes), min(box[1] for box in boxes)
            right, top = max(box[2] for box in boxes), max(box[3] for box in boxes)
            corners = [posconv.to_bitmap(x, y) for x, y in [(left, bottom), (left, top), (right, bottom), (right, top)]]
            bbox = {"x_min": min(point[0] for point in corners), "y_min": min(point[1] for point in corners), "x_max": max(point[0] for point in corners), "y_max": max(point[1] for point in corners)}
            tokens.append({"value": value, "original_value": value, "score": None, "source": "pdf_text", "bbox": bbox})

        characters.clear()
        boxes.clear()

    for index in range(textpage.count_chars()):
        codepoint = pdfium.raw.FPDFText_GetUnicode(textpage, index)

        if not codepoint:
            continue

        character = chr(codepoint)

        if character in "\r\n":
            # PDFium이 같은 줄의 글자 사이에도 줄바꿈을 삽입할 수 있어 좌표로 확인한다.
            line_break = True
            continue

        if character == "\t":
            finish_line()
            continue

        if character.isspace():
            if characters:
                characters.append(character)

            continue

        left, bottom, right, top = textpage.get_charbox(index)

        if right <= left or top <= bottom:
            continue

        font_box = textpage.get_charbox(index, loose=True)

        if boxes:
            previous = boxes[-1]
            line_bottom, line_top = min(box[1] for box in boxes), max(box[3] for box in boxes)
            line_height = line_top - line_bottom
            height = top - bottom
            font_height = min(previous_font[3] - previous_font[1], font_box[3] - font_box[1])
            overlap = min(previous_font[3], font_box[3]) - max(previous_font[1], font_box[1])
            gap = left - previous[2]

            if overlap < font_height * 0.5 or gap > max(height, line_height) * 1.5 or right < previous[0]:
                finish_line()
            elif line_break and gap > max(height, line_height) * 0.25 and not characters[-1].isspace():
                characters.append(" ")

        characters.append(character)
        boxes.append((left, bottom, right, top))
        line_break = False
        previous_font = font_box

    finish_line()

    return tokens


def iter_pages(source, file_type):
    """Yield (index, RGB image, native tokens, native_only) one page at a time.
    Convert PDF text to pixel coordinates; native_only requires text and no image objects."""
    source = resolve_source_path(source)

    if file_type != "application/pdf":
        with Image.open(source) as image:
            yield 0, image.convert("RGB"), [], False

        return

    with pdfium.PdfDocument(source) as pdf:
        for index in range(len(pdf)):
            with closing(pdf[index]) as page:
                width, height = page.get_size()
                scale = min(2.5, 2200 / max(width, height))
                bitmap = page.render(scale=scale)

                try:
                    image = bitmap.to_pil().convert("RGB")
                    posconv = bitmap.get_posconv(page)
                finally:
                    bitmap.close()

                if image.width * image.height > MAX_PAGE_PIXELS:
                    raise ValueError("The rendered PDF page is too large.")

                # Use rendered coordinates to account for PDF rotation and pixel rounding.
                with closing(page.get_textpage()) as textpage:
                    tokens = pdf_text_tokens(textpage, posconv)

                has_images = any(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]))

                yield index, image, tokens, bool(tokens) and not has_images
