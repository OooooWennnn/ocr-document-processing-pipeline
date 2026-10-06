"""Refine predicted cells using table rules in the source image."""

import math
from src.schemas.schema import BBox, Cell


def refine_cell_geometry(image, cells):
    """Use a closed grid, or snap edges within 4px and split on rules covering 90% of cell width.
    Return new cells without changing the image or input cells."""
    import cv2
    import numpy as np

    ruled_cells = recover_ruled_cells(image, cells)

    # Prefer a clear source grid even when predicted cells do not overlap.
    # Keep the model fallback when partial rules recover fewer cells.
    if ruled_cells and len(ruled_cells) >= len(cells):
        return ruled_cells

    ink = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    image_height, image_width = image.shape[:2]

    def edge(mask, position, start, end, vertical_axis=False):
        """Find the closest supported rule near an edge, or keep the predicted coordinate."""
        a = max(0, math.ceil(start) + 2)
        b = min(image_height if vertical_axis else image_width, math.floor(end) - 2)

        if b - a < 8:
            return position

        candidates = []

        for n in range(max(0, round(position) - 4), min(image_width if vertical_axis else image_height, round(position) + 5)):
            segment = mask[a:b, n] if vertical_axis else mask[n, a:b]
            strength = float(np.mean(segment > 0))

            if strength >= 0.85:
                candidates.append(n)

        return float(min(candidates, key=lambda n: abs(n - position))) if candidates else position

    refined_cells = []

    for cell in cells:
        b = cell.bbox
        x1 = edge(vertical, b.x_min, b.y_min, b.y_max, True)
        x2 = edge(vertical, b.x_max, b.y_min, b.y_max, True)
        y1 = edge(horizontal, b.y_min, x1, x2)
        y2 = edge(horizontal, b.y_max, x1, x2)

        if x2 - x1 < 8 or y2 - y1 < 8:
            refined_cells.append(cell)
            continue

        left = max(0, math.ceil(x1) + 2)
        right = min(image_width, math.floor(x2) - 2)
        split_lines = []

        if right - left >= 8:
            for y in range(max(0, math.ceil(y1) + 5), min(image_height, math.floor(y2) - 4)):
                if float(np.mean(horizontal[y, left:right] > 0)) >= 0.9:
                    if not split_lines or y - split_lines[-1] > 3:
                        split_lines.append(y)

        y_boundaries = [y1, *split_lines, y2]

        if len(y_boundaries) > 2 and min(v - u for u, v in zip(y_boundaries, y_boundaries[1:])) < 8:
            y_boundaries = [y1, y2]

        refined_cells.extend(Cell(-1, -1, BBox(x1, top, x2, bottom)) for top, bottom in zip(y_boundaries, y_boundaries[1:]))

    return refined_cells


def recover_ruled_cells(image, cells):
    """Recover cells from closed multi-row, multi-column rules near predicted cells.
    Keep merged gaps, preserve inputs and return [] when the rules are insufficient."""
    import cv2
    import numpy as np

    if not cells:
        return []

    x_min = min(cell.bbox.x_min for cell in cells)
    y_min = min(cell.bbox.y_min for cell in cells)
    x_max = max(cell.bbox.x_max for cell in cells)
    y_max = max(cell.bbox.y_max for cell in cells)
    bounds = BBox(x_min, y_min, x_max, y_max)
    padding = max(8, min(32, round(min(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min) * 0.03)))
    left = max(0, math.floor(bounds.x_min) - padding)
    top = max(0, math.floor(bounds.y_min) - padding)
    right = min(image.shape[1], math.ceil(bounds.x_max) + padding)
    bottom = min(image.shape[0], math.ceil(bounds.y_max) + padding)

    if right - left < 30 or bottom - top < 30:
        return []

    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    # Detect faint rules relative to their local background.
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 6)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, gray.shape[1] // 20), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, gray.shape[0] // 20))))

    def line_centers(indices, strength=None):
        """Group adjacent rule pixels and prefer the strongest column position across rows."""
        groups = []

        for position in indices:
            if groups and position - groups[-1][-1] <= 3:
                groups[-1].append(position)
            else:
                groups.append([position])

        centers = []

        for group in groups:
            if strength is not None:
                peak = max(strength[position] for position in group)
                group = [position for position in group if strength[position] == peak]

            centers.append(int(round(float(np.mean(group)))))

        return centers

    y_lines = line_centers(np.flatnonzero(np.mean(horizontal > 0, axis=1) >= 0.7))

    if len(y_lines) < 3 or min(b - a for a, b in zip(y_lines, y_lines[1:])) < 8:
        return []

    # Column rules may appear only below a merged header.
    # Check vertical continuity per row, not across the whole table.
    column_strength = np.zeros(gray.shape[1])

    for y1, y2 in zip(y_lines, y_lines[1:]):
        column_strength = np.maximum(column_strength, np.mean(vertical[y1 + 3:y2 - 2] > 0, axis=0))

    x_lines = line_centers(np.flatnonzero(column_strength >= 0.8), np.mean(vertical > 0, axis=0))

    if len(x_lines) < 3:
        return []

    if min(b - a for a, b in zip(x_lines, x_lines[1:])) < 8:
        return []

    recovered = []

    for y1, y2 in zip(y_lines, y_lines[1:]):
        boundaries = []

        for x in x_lines:
            segment = vertical[y1 + 3:y2 - 2, max(0, x - 2):x + 3]

            if float(np.mean(np.any(segment > 0, axis=1))) >= 0.8:
                boundaries.append(x)

        # Reject open rows rather than inventing missing borders.
        if not boundaries or boundaries[0] != x_lines[0] or boundaries[-1] != x_lines[-1]:
            return []

        for x1, x2 in zip(boundaries, boundaries[1:]):
            for y in (y1, y2):
                segment = horizontal[max(0, y - 2):y + 3, x1 + 3:x2 - 2]

                if float(np.mean(np.any(segment > 0, axis=0))) < 0.8:
                    return []

            recovered.append(Cell(-1, -1, BBox(left + x1, top + y1, left + x2, top + y2)))

    return recovered
