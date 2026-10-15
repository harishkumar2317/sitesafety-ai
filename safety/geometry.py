"""Bounding-box geometry helpers (pure Python / NumPy free)."""

Box = tuple  # (x1, y1, x2, y2)


def area(box: Box) -> float:
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def iou(a: Box, b: Box) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center(box: Box) -> tuple:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_inside(inner: Box, outer: Box) -> bool:
    cx, cy = center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def expand(box: Box, ratio: float, frame_shape=None) -> Box:
    w = box[2] - box[0]
    h = box[3] - box[1]
    dx = w * ratio
    dy = h * ratio
    out = (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)
    if frame_shape is None:
        return out
    fh, fw = frame_shape[:2]
    return (max(0, out[0]), max(0, out[1]), min(fw - 1, out[2]), min(fh - 1, out[3]))


def sub_box(box: Box, y0: float, y1: float, x0: float = 0.05, x1: float = 0.95) -> Box:
    """Slice a relative window out of a box: y0/y1/x0/x1 are 0..1 fractions."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (
        box[0] + w * x0,
        box[1] + h * y0,
        box[0] + w * x1,
        box[1] + h * y1,
    )


def clip(box: Box, frame_shape) -> Box:
    fh, fw = frame_shape[:2]
    return (
        max(0, min(fw - 1, box[0])),
        max(0, min(fh - 1, box[1])),
        max(0, min(fw - 1, box[2])),
        max(0, min(fh - 1, box[3])),
    )


def to_int(box: Box) -> tuple:
    return tuple(int(round(v)) for v in box)


def size(box: Box) -> tuple:
    return (box[2] - box[0], box[3] - box[1])


def distance(a: Box, b: Box) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
