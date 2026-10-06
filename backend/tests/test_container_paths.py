import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.services.document_input import resolve_source_path


class ContainerPathTests(unittest.TestCase):
    def test_old_storage_path_maps_without_database_change(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"OCR_STORAGE_DIR": root, "OCR_STORAGE_LEGACY_ROOT": "/old/project/storage"}):
            source = "/old/project/storage/uploads/example.png"

            self.assertEqual(resolve_source_path(source), Path(root) / "uploads/example.png")

            unrelated = "/other/project/example.png"

            self.assertEqual(resolve_source_path(unrelated), Path(unrelated))

            traversal = "/old/project/storage/../outside.png"

            self.assertEqual(resolve_source_path(traversal), Path(traversal))

    def test_existing_file_is_used_as_is(self):
        with tempfile.NamedTemporaryFile() as source, patch.dict(os.environ, {"OCR_STORAGE_DIR": "/app/storage", "OCR_STORAGE_LEGACY_ROOT": "/old/project/storage"}):
            self.assertEqual(resolve_source_path(source.name), Path(source.name))
