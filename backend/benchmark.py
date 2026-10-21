"""Run real OCR measurements inside the worker image; not part of unittest discovery."""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import resource
import time
from pathlib import Path
from unittest.mock import patch

import cv2
from src.services.table_extraction import TableExtraction


@contextmanager
def measure_models(timings):
    """PaddleX 모델별 호출 시간과 결과 수를 기록한다. 추출 결과나 실행 설정은 바꾸지 않는다."""
    from paddlex.inference.models.base.predictor.base_predictor import BasePredictor

    original_call = BasePredictor.__call__

    def measured_call(model, *args, **kwargs):
        """결과를 계산하는 시간만 합산하고, 호출자가 결과를 처리하는 시간은 제외한다."""
        metric = timings.setdefault(model.model_name, {"seconds": 0.0, "calls": 0, "items": 0})
        metric["calls"] += 1
        iterator = iter(original_call(model, *args, **kwargs))

        while True:
            started = time.perf_counter()

            try:
                result = next(iterator)
            except StopIteration:
                break
            finally:
                metric["seconds"] += time.perf_counter() - started

            metric["items"] += 1
            yield result

    with patch.object(BasePredictor, "__call__", measured_call):
        yield


def cell_iou(expected, actual):
    """정답 셀과 추출된 셀의 면적 겹침 비율(IoU)을 계산한다."""
    x1, y1, x2, y2 = expected
    width = max(0, min(x2, actual["x_max"]) - max(x1, actual["x_min"]))
    height = max(0, min(y2, actual["y_max"]) - max(y1, actual["y_min"]))
    intersection = width * height
    actual_area = (actual["x_max"] - actual["x_min"]) * (actual["y_max"] - actual["y_min"])

    return intersection / ((x2 - x1) * (y2 - y1) + actual_area - intersection)


def check_cells(page, grid):
    """수동 정답 영역의 셀 위치·값과 페이지 텍스트를 검사한다. confidence는 정답으로 쓰지 않는다."""
    cells = [cell for table in page["tables"] for row in table["table_rows"] for cell in row["cells"]]
    result = {"labelled": 0, "located": 0, "exact": 0, "nonempty_labelled": 0, "nonempty_exact": 0, "text_exact": 0, "failures": [], "text_failures": []}

    for row_index, row in enumerate(grid.get("rows", [])):
        for column_index, expected in enumerate(row):
            xs, ys = grid["x_edges"], grid["y_edges"]
            box = [xs[column_index], ys[row_index], xs[column_index + 1], ys[row_index + 1]]
            cell = max(cells, key=lambda item: cell_iou(box, item["bbox"]), default=None)
            located = cell is not None and cell_iou(box, cell["bbox"]) >= 0.5
            actual = cell["original_value"] if located else None
            exact = located and " ".join(actual.casefold().split()) == " ".join(expected.casefold().split())
            result["labelled"] += 1
            result["located"] += int(located)
            result["exact"] += int(exact)
            result["nonempty_labelled"] += int(bool(expected.strip()))
            result["nonempty_exact"] += int(exact and bool(expected.strip()))

            if not exact:
                result["failures"].append({"bbox": box, "expected": expected, "actual": actual})

            x1, y1, x2, y2 = box
            tokens = [text for text in page["texts"] if x1 <= (text["bbox"]["x_min"] + text["bbox"]["x_max"]) / 2 < x2 and y1 <= (text["bbox"]["y_min"] + text["bbox"]["y_max"]) / 2 < y2]
            text_value = " ".join(text["value"] for text in sorted(tokens, key=lambda text: text["bbox"]["x_min"]))
            text_exact = " ".join(text_value.casefold().split()) == " ".join(expected.casefold().split())
            result["text_exact"] += int(text_exact)

            if not text_exact:
                result["text_failures"].append({"bbox": box, "expected": expected, "actual": text_value})

    return result


def compare_page(page, original):
    """정답 영역 밖의 글자·좌표와 빈 셀·병합 구조도 이전 결과와 비교한다. confidence는 제외한다."""
    def texts(document):
        return [(text["value"], text["bbox"]) for text in document["texts"]]

    def cells(document):
        keys = ["row", "col", "rowspan", "colspan", "bbox", "original_value"]
        tables = []

        for table in document["tables"]:
            rows = []

            for row in table["table_rows"]:
                rows.append([{key: cell[key] for key in keys} for cell in row["cells"]])

            tables.append(rows)

        return tables

    def boundaries(document):
        return [(table["bbox"], table["x_lines"], table["y_lines"]) for table in document["tables"]]

    return {
        "texts_and_boxes_equal": texts(page) == texts(original),
        "cells_and_boxes_equal": cells(page) == cells(original),
        "table_boundaries_equal": boundaries(page) == boundaries(original),
        "warnings_equal": page["warnings"] == original["warnings"],
    }


def main():
    """같은 샘플을 순서대로 처리하고 시간·CPU·메모리·수동 정답 비교를 JSON 파일에 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-models", action="store_true", help="Record PaddleX model times without changing predictions.")
    parser.add_argument("--baseline", type=Path, help="Compare complete text, coordinates and tables with an earlier output.")
    args = parser.parse_args()
    samples = json.loads(args.manifest.read_text())
    baseline = None
    comparison_failed = False

    if args.baseline:
        previous = json.loads(args.baseline.read_text())
        baseline = {run["metrics"]["sample"]: run["page"] for run in previous["runs"]}

    started = time.perf_counter()
    extractor = TableExtraction()
    output = {
        "model_load_seconds": time.perf_counter() - started,
        "model_load_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "memory_scope": "Process high-water RSS, including initialization and all preceding pages.",
        "runs": [],
    }

    for sample in samples:
        source = args.input_dir / sample["file"]

        if hashlib.sha256(source.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"Sample hash mismatch: {sample['id']}")

        image = cv2.imread(str(source))

        if image is None:
            raise ValueError(f"Cannot read image: {source}")

        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        model_timings = {}
        started = time.perf_counter()

        measurement = measure_models(model_timings) if args.profile_models else nullcontext()

        with measurement:
            _, page = extractor.detect_document(image)
        elapsed = time.perf_counter() - started
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)

        metrics = {
            "sample": sample["id"],
            "seconds": round(elapsed, 3),
            "cpu_seconds": round(cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime, 3),
            "process_peak_rss_mib": round(cpu_after.ru_maxrss / 1024, 1),
            "text_count": len(page["texts"]),
            "table_count": len(page["tables"]),
            "cell_count": sum(len(row["cells"]) for table in page["tables"] for row in table["table_rows"]),
            "quality": check_cells(page, sample.get("grid", {})),
        }

        if args.profile_models:
            metrics["models"] = {name: {**values, "seconds": round(values["seconds"], 3)} for name, values in model_timings.items()}

        if baseline is not None:
            metrics["baseline_comparison"] = compare_page(page, baseline[sample["id"]])

            if not all(metrics["baseline_comparison"].values()):
                comparison_failed = True

        output["runs"].append({"metrics": metrics, "page": page})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    return 1 if comparison_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
