import math
from collections.abc import Iterator

from PIL import Image

MESH_PX = 8
TOLERANCE_PX = 0.5

Point = tuple[float, float]
Box = tuple[int, int, int, int]


def perspective(image: Image.Image, yaw: float, pitch: float, fov: float, size: tuple[int, int]) -> Image.Image:
    width, height = size
    cols, rows = image.size
    focal = 0.5 * width / math.tan(math.radians(fov) / 2)
    turn, tilt = math.radians(yaw), math.radians(pitch)

    def source(x: float, y: float, anchor: float) -> Point:
        right, down, ahead = (x - width / 2) / focal, (y - height / 2) / focal, 1.0
        down, ahead = down * math.cos(tilt) - ahead * math.sin(tilt), down * math.sin(tilt) + ahead * math.cos(tilt)
        right, ahead = right * math.cos(turn) + ahead * math.sin(turn), ahead * math.cos(turn) - right * math.sin(turn)
        lng, lat = math.atan2(right, ahead), math.atan2(down, math.hypot(right, ahead))
        column = (lng / math.tau + 0.5) * (cols - 1) + 0.5
        return column + cols * round((anchor - column) / cols), (lat / math.pi + 0.5) * (rows - 1) + 0.5

    def quads(box: Box) -> Iterator[tuple[Box, list[Point]]]:
        left, top, right, bottom = box
        anchor = source(left, top, cols / 2)[0]
        corners = [source(x, y, anchor) for x, y in ((left, top), (left, bottom), (right, bottom), (right, top))]
        mid = source((left + right) / 2, (top + bottom) / 2, anchor)
        miss = math.dist(mid, (sum(x for x, _ in corners) / 4, sum(y for _, y in corners) / 4))
        if miss <= TOLERANCE_PX or (right - left, bottom - top) == (1, 1):
            yield box, corners
            return
        xs = sorted({left, (left + right) // 2, right})
        ys = sorted({top, (top + bottom) // 2, bottom})
        for x0, x1 in zip(xs, xs[1:]):
            for y0, y1 in zip(ys, ys[1:]):
                yield from quads((x0, y0, x1, y1))

    cells = [(x, y, min(x + MESH_PX, width), min(y + MESH_PX, height)) for x in range(0, width, MESH_PX) for y in range(0, height, MESH_PX)]
    mesh = [quad for cell in cells for quad in quads(cell)]
    pad = math.ceil(max(0.0, *(max(-x, x - cols) for _, corners in mesh for x, _ in corners))) + 2
    wrapped = Image.new(image.mode, (cols + 2 * pad, rows))
    for offset in (-cols, 0, cols):
        wrapped.paste(image, (pad + offset, 0))
    data = [(box, tuple(v for x, y in corners for v in (x + pad, y))) for box, corners in mesh]
    return wrapped.transform(size, Image.Transform.MESH, data, resample=Image.Resampling.BICUBIC)
