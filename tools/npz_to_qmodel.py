"""Convert a portfolio NPZ fixture to a little-endian version 1 qmodel file.

Run with: uv run --no-project --with numpy python -I tools/npz_to_qmodel.py IN OUT
"""

import argparse
import struct
from pathlib import Path

import numpy as np


def convert(source: Path, output: Path) -> None:
    """Preserve f64 coefficients and offset, with canonical sparse endpoints."""
    with np.load(source, allow_pickle=False) as model:
        h = np.asarray(model["h"], dtype="<f8")
        edges = model["edges"]
        j = np.asarray(model["j"], dtype="<f8")
        offset = np.asarray(model["offset"], dtype="<f8")
        if h.ndim != 1 or j.ndim != 1 or edges.shape != (j.size, 2):
            raise ValueError("expected h[n], edges[m,2], and j[m]")
        if not np.issubdtype(edges.dtype, np.integer):
            raise ValueError("edge endpoints must be integers")
        if offset.size != 1:
            raise ValueError("offset must be a scalar")
        if not all(np.isfinite(values).all() for values in (h, j, offset)):
            raise ValueError("coefficients and offset must be finite")
        if np.any(edges < 0) or np.any(edges >= h.size):
            raise ValueError("edge endpoints must be within h")
        if np.any(edges > np.iinfo(np.uint32).max):
            raise ValueError("edge endpoints must fit u32")
        edges = np.sort(edges, axis=1)
        if np.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("self-loops are not supported")
        records = np.empty(j.size, dtype=[("i", "<u4"), ("j", "<u4"), ("value", "<f8")])
        records["i"] = edges[:, 0]
        records["j"] = edges[:, 1]
        records["value"] = j
        with output.open("wb") as stream:
            stream.write(
                struct.pack("<4sIIQQd", b"QMDL", 1, 0, h.size, j.size, offset.item())
            )
            stream.write(h.tobytes())
            stream.write(records.tobytes())


def main() -> None:
    """Read two positional paths and report conversion errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        convert(args.source, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{args.source}: {error}\n")


if __name__ == "__main__":
    main()
