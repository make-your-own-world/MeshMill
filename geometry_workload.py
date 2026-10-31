"""In-memory geometry workload receipts, independent of rendering and disk I/O."""
from threading import Lock

_lock = Lock()
_totals = {device: [0, 0, 0] for device in ("CPU", "GPU")}


def record(device, points, faces):
    receipt = [len(points), len(faces), int(points.nbytes + faces.nbytes)]
    with _lock:
        _totals[device] = [a + b for a, b in zip(_totals[device], receipt)]


def snapshot():
    with _lock:
        return {device: list(values) for device, values in _totals.items()}


def difference(after, before):
    return {device: [a - b for a, b in zip(after[device], before[device])]
            for device in after}


def merge(receipts):
    with _lock:
        for device, values in receipts.items():
            if device not in _totals or len(values) != 3 or any(v < 0 for v in values):
                raise ValueError("Invalid geometry workload receipt")
            _totals[device] = [a + int(b) for a, b in zip(_totals[device], values)]
