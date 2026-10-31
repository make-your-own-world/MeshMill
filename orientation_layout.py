"""Continuous perimeter layout for the orientation cube's value badges."""

import numpy as np


def place_axis_badges(desired, cube_bounds, width, height, badge_width, badge_height):
    center = np.array((width / 2, height / 2))
    half = np.array((badge_width, badge_height), dtype=float) / 2
    radius = np.array((width, height)) / 2 - half - 4
    desired = np.asarray(desired) - center
    # Map axis directions continuously onto the surrounding rectangular rail.
    projected = desired / np.maximum(np.max(np.abs(desired) / radius, axis=1), 1e-9)[:, None]
    rx, ry = radius
    perimeter = 4 * (rx + ry)
    positions = []
    for x, y in projected:
        if abs(y + ry) < 1e-6:
            value = x + rx
        elif abs(x - rx) < 1e-6:
            value = 2 * rx + y + ry
        elif abs(y - ry) < 1e-6:
            value = 4 * rx + 2 * ry - x - rx
        else:
            value = perimeter - y - ry
        positions.append(value)
    positions = np.asarray(positions)
    gap = badge_width + badge_height + 8
    for _ in range(12):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            delta = (positions[second] - positions[first] + perimeter / 2) % perimeter - perimeter / 2
            if abs(delta) < gap:
                correction = (gap - abs(delta)) / 2
                sign = 1 if delta >= 0 else -1
                positions[first] -= sign * correction
                positions[second] += sign * correction
    result = []
    for value in positions % perimeter:
        if value < 2 * rx:
            point = (-rx + value, -ry)
        elif value < 2 * rx + 2 * ry:
            point = (rx, -ry + value - 2 * rx)
        elif value < 4 * rx + 2 * ry:
            point = (rx - (value - 2 * rx - 2 * ry), ry)
        else:
            point = (-rx, ry - (value - 4 * rx - 2 * ry))
        result.append(center + point)
    return np.asarray(result)
