"""A minimal, vectorised ESRI shapefile reader.

The Belgian OSM extract has 7.6 million building polygons in a 1.4 GB `.shp`,
of which a few tens of thousands fall inside the Antwerp measurement area.
Walking all of them through a general-purpose reader costs minutes and a lot of
memory; but every polygon record stores its own bounding box at a fixed offset
from the start of the record, and the `.shx` index gives every record's offset
in one array.  So the bounding boxes can be gathered with a single fancy-index
into a memory map, filtered with numpy, and only the surviving records parsed.

Only what this study needs is implemented: polygon and polyline geometry, and
one character field from the `.dbf`.
"""
from __future__ import annotations
import numpy as np


def read_shx(path: str) -> np.ndarray:
    """Byte offset of every record in the companion `.shp`."""
    raw = np.fromfile(path, dtype=">i4", offset=100)
    return raw[0::2].astype(np.int64) * 2          # 16-bit words -> bytes


def bboxes(shp_path: str, offsets: np.ndarray) -> np.ndarray:
    """(N, 4) array of xmin, ymin, xmax, ymax, one row per record."""
    mm = np.memmap(shp_path, dtype=np.uint8, mode="r")
    base = offsets + 12                       # 8 record header + 4 shape type
    idx = base[:, None] + np.arange(32)[None, :]
    raw = np.ascontiguousarray(mm[idx])
    return raw.view(np.float64).reshape(-1, 4)


class Shapes:
    """Lazy access to individual records."""

    def __init__(self, shp_path: str, offsets: np.ndarray):
        self.mm = np.memmap(shp_path, dtype=np.uint8, mode="r")
        self.off = offsets

    def parts_points(self, i: int):
        o = int(self.off[i]) + 12 + 32
        n_parts, n_pts = np.frombuffer(self.mm[o: o + 8].tobytes(), dtype="<i4")
        o += 8
        parts = np.frombuffer(self.mm[o: o + 4 * n_parts].tobytes(), dtype="<i4")
        o += 4 * n_parts
        pts = np.frombuffer(self.mm[o: o + 16 * n_pts].tobytes(),
                            dtype="<f8").reshape(-1, 2)
        return parts, pts


def dbf_field(path: str, field: str, idx: np.ndarray) -> list[str]:
    """Read one character column of a `.dbf`, for selected record indices."""
    with open(path, "rb") as f:
        head = f.read(32)
        header_len = int(np.frombuffer(head[8:10], "<u2")[0])
        rec_len = int(np.frombuffer(head[10:12], "<u2")[0])
        f.seek(32)
        pos, off, width = 0, None, None
        while True:
            fd = f.read(32)
            if not fd or fd[0] == 0x0D:
                break
            name = fd[:11].split(b"\x00")[0].decode("latin-1")
            w = fd[16]
            if name == field:
                off, width = pos, w
            pos += w
        if off is None:
            raise KeyError(field)
        out = []
        for i in idx:
            f.seek(header_len + int(i) * rec_len + 1 + off)
            out.append(f.read(width).decode("latin-1").strip())
    return out
