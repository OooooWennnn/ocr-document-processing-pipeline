"""Version-specific chart initialization workaround; review when upgrading PaddleX."""

from contextlib import contextmanager
from importlib.metadata import version
import logging
from threading import RLock
from unittest.mock import patch

_lock = RLock()


@contextmanager
def without_unused_chart():
    """Skip disabled chart initialization in PaddleX 3.4.1 only; restore the original method on exit."""
    if version("paddlex") != "3.4.1":
        logging.getLogger(__name__).warning("PaddleX version changed; skipping the chart-load optimization. Revalidate adapter.")
        yield

        return

    from paddlex.inference.pipelines.layout_parsing.pipeline_v2 import _LayoutParsingPipelineV2

    original = _LayoutParsingPipelineV2.create_model

    def create_model(pipeline, config, *args, **kwargs):
        """Skip the disabled chart model and delegate all other model creation."""
        if config.get("module_name") == "chart_recognition" and not pipeline.use_chart_recognition:
            logging.getLogger(__name__).info("Skipping disabled PP-Chart2Table model")

            return None

        return original(pipeline, config, *args, **kwargs)

    # Serialize the patch and restore the factory even if initialization fails.
    with _lock, patch.object(_LayoutParsingPipelineV2, "create_model", create_model):
        yield
