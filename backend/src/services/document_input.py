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

                tokens = []

                # Use rendered coordinates to account for PDF rotation and pixel rounding.
                with closing(page.get_textpage()) as textpage:
                    for i in range(textpage.count_rects()):
                        left, bottom, right, top = textpage.get_rect(i)
                        value = textpage.get_text_bounded(left, bottom, right, top).strip()

                        if value and right > left and top > bottom:
                            corners = [posconv.to_bitmap(x, y) for x, y in [(left, bottom), (left, top), (right, bottom), (right, top)]]

                            bbox = {
                                "x_min": min(corner[0] for corner in corners),
                                "y_min": min(corner[1] for corner in corners),
                                "x_max": max(corner[0] for corner in corners),
                                "y_max": max(corner[1] for corner in corners),
                            }
                            token = {
                                "value": value,
                                "original_value": value,
                                "score": None,
                                "source": "pdf_text",
                                "bbox": bbox,
                            }
                            tokens.append(token)

                has_images = any(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]))

                yield index, image, tokens, bool(tokens) and not has_images
