"""
data_parser.py — Parse SUN RGB-D scenes into structured JSON records.

SUN RGB-D folder layout expected under --raw-root:
    <raw_root>/
        SUNRGBD/
            kv1/   kv2/   realsense/   xtion/
                <device_subfolder>/
                    <scene_folder>/
                        scene.txt
                        image/          <- RGB images (*.jpg)
                        annotation2Dfinal/
                            index.json  <- object polygon annotations
                        annotation3Dlayout/
                            layout.mat  <- room box (optional)
        SUNRGBDMeta2DBB_v2.mat          <- top-level metadata (optional supplement)

Each output record conforms to:
    {
        "scene_id":    str,
        "room_type":   str,
        "objects":     [str, ...],
        "num_objects": int,
        "layout_dims": {"width_m": float, "depth_m": float, "height_m": float} | null,
        "rgb_relpath": str
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sensor sub-folder names directly under SUNRGBD/
SENSOR_DIRS = ("kv1", "kv2", "realsense", "xtion")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class LayoutDims:
    width_m: float
    depth_m: float
    height_m: float


@dataclass
class SceneRecord:
    scene_id: str
    room_type: str
    objects: list[str]
    num_objects: int
    layout_dims: Optional[LayoutDims]
    rgb_relpath: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict, with layout_dims as dict or None."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-scene file readers
# ---------------------------------------------------------------------------

def _read_room_type(scene_dir: Path) -> str:
    """Read scene.txt and return a normalised snake_case room type string.

    Takes the first non-empty line, lowercases it, and replaces spaces/hyphens
    with underscores.  Returns ``"unknown"`` if the file is absent or empty.

    Args:
        scene_dir: Path to a single scene folder.

    Returns:
        Normalised room type, e.g. ``"living_room"``.
    """
    scene_txt = scene_dir / "scene.txt"
    if not scene_txt.is_file():
        logger.warning("scene.txt missing: %s", scene_dir)
        return "unknown"

    raw = scene_txt.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        logger.warning("scene.txt empty: %s", scene_dir)
        return "unknown"

    first_line = raw.splitlines()[0].strip().lower()
    return re.sub(r"[\s\-]+", "_", first_line)


def _read_objects(scene_dir: Path) -> list[str]:
    """Parse annotation2Dfinal/index.json and return deduplicated object labels.

    Handles two JSON layouts seen in the wild:
    - A list of annotation dicts (each has ``"name"`` or ``"label"``).
    - A dict with an ``"annotation"`` or ``"objects"`` key containing that list.

    Labels are lowercased and stripped.  Returns ``[]`` on any failure.

    Args:
        scene_dir: Path to a single scene folder.

    Returns:
        Sorted, deduplicated list of object label strings.
    """
    index_json = scene_dir / "annotation2Dfinal" / "index.json"
    if not index_json.is_file():
        logger.debug("annotation2Dfinal/index.json missing: %s", scene_dir)
        return []

    try:
        data = json.loads(index_json.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        logger.warning("Malformed index.json in %s: %s", scene_dir, exc)
        return []

    labels: list[str] = []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("annotation") or data.get("objects") or []
    else:
        logger.warning("Unrecognised index.json structure in %s", scene_dir)
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("label") or item.get("class")
        if name and isinstance(name, str):
            labels.append(name.strip().lower())

    return sorted(set(labels))


def _read_layout_dims(scene_dir: Path) -> Optional[LayoutDims]:
    """Extract room box dimensions from annotation3Dlayout/layout.mat.

    The mat file stores 8 corner points of the room bounding box in metres
    (camera coordinate system: X=right, Y=down, Z=forward).  Width, height,
    and depth are recovered as axis-aligned extents across those corners.

    Returns ``None`` on any failure (file missing, scipy unavailable, unexpected
    array shape) so callers are never blocked by optional layout data.

    Args:
        scene_dir: Path to a single scene folder.

    Returns:
        :class:`LayoutDims` with rounded metric values, or ``None``.
    """
    layout_mat = scene_dir / "annotation3Dlayout" / "layout.mat"
    if not layout_mat.is_file():
        logger.debug("annotation3Dlayout/layout.mat missing: %s", scene_dir)
        return None

    try:
        import scipy.io  # deferred — not always installed
        mat = scipy.io.loadmat(
            str(layout_mat),
            squeeze_me=True,
            struct_as_record=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot load layout.mat in %s: %s", scene_dir, exc)
        return None

    # Prefer known variable names, then fall back to the first array key.
    corners: Optional[np.ndarray] = None
    for key in ("layout", "corners", "pts"):
        if key in mat:
            corners = np.asarray(mat[key], dtype=float)
            break

    if corners is None:
        array_keys = [k for k in mat if not k.startswith("__")]
        if not array_keys:
            logger.warning("No usable arrays in layout.mat: %s", scene_dir)
            return None
        corners = np.asarray(mat[array_keys[0]], dtype=float)

    # Normalise to shape (N, 3) regardless of whether it came in as (8,3) or (3,8)
    if corners.ndim != 2 or 3 not in corners.shape:
        logger.warning(
            "Unexpected layout.mat shape %s in %s", corners.shape, scene_dir
        )
        return None

    if corners.shape[1] != 3:
        corners = corners.T

    extents = corners.max(axis=0) - corners.min(axis=0)  # (3,) in metres
    return LayoutDims(
        width_m=round(float(extents[0]), 3),
        depth_m=round(float(extents[2]), 3),
        height_m=round(float(extents[1]), 3),
    )


def _find_rgb_image(scene_dir: Path, raw_root: Path) -> Optional[str]:
    """Return the relative path (from raw_root) to the first RGB image in image/.

    Searches for .jpg, .jpeg, .png in priority order.  Returns ``None`` if the
    image/ sub-directory is absent or contains no recognised image files.

    Args:
        scene_dir: Path to a single scene folder.
        raw_root: Root of the raw data directory (used to compute relative path).

    Returns:
        Relative path string or ``None``.
    """
    image_dir = scene_dir / "image"
    if not image_dir.is_dir():
        logger.debug("image/ directory missing: %s", scene_dir)
        return None

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return str(matches[0].relative_to(raw_root))

    logger.debug("No RGB image found under image/: %s", scene_dir)
    return None


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _discover_scene_dirs(sunrgbd_root: Path) -> list[Path]:
    """Walk the four sensor sub-trees and collect leaf scene directories.

    A directory is classified as a scene leaf when it contains ``scene.txt``.
    Once a leaf is found we stop descending into it so nested structures are
    not double-counted.

    Args:
        sunrgbd_root: Path to the top-level SUNRGBD/ folder.

    Returns:
        Sorted list of scene directory paths.
    """
    scene_dirs: list[Path] = []
    for sensor in SENSOR_DIRS:
        sensor_path = sunrgbd_root / sensor
        if not sensor_path.is_dir():
            logger.debug("Sensor directory not found, skipping: %s", sensor_path)
            continue
        for root, dirs, files in os.walk(sensor_path):
            if "scene.txt" in files:
                scene_dirs.append(Path(root))
                dirs.clear()  # do not recurse into scene sub-folders

    logger.info("Discovered %d scene directories.", len(scene_dirs))
    return sorted(scene_dirs)


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class DataParser:
    """Parse SUN RGB-D scenes from *raw_root* into a list of :class:`SceneRecord`.

    Args:
        raw_root: Path to the raw data root containing the ``SUNRGBD/``
            sub-directory and optionally ``SUNRGBDMeta2DBB_v2.mat``.
        limit: If set, stop after processing this many scenes.  ``None``
            processes all discovered scenes.
    """

    def __init__(self, raw_root: str | Path, limit: Optional[int] = None) -> None:
        self.raw_root = Path(raw_root).resolve()
        self.limit = limit
        self._sunrgbd_root = self.raw_root / "SUNRGBD"

    def load(self) -> list[SceneRecord]:
        """Discover and parse all scenes, returning validated records.

        Scenes that cannot be parsed (e.g. no RGB image) are skipped with a
        warning rather than raising.

        Returns:
            List of :class:`SceneRecord` instances, one per valid scene.

        Raises:
            FileNotFoundError: If ``SUNRGBD/`` does not exist under raw_root.
        """
        if not self._sunrgbd_root.is_dir():
            raise FileNotFoundError(
                f"SUNRGBD directory not found under raw_root: {self._sunrgbd_root}"
            )

        scene_dirs = _discover_scene_dirs(self._sunrgbd_root)
        if self.limit is not None:
            scene_dirs = scene_dirs[: self.limit]
            logger.info("Limiting to first %d scenes.", self.limit)

        records: list[SceneRecord] = []
        skipped = 0

        for scene_dir in scene_dirs:
            record = self._parse_scene(scene_dir)
            if record is None:
                skipped += 1
            else:
                records.append(record)

        logger.info(
            "Parsed %d scenes successfully; skipped %d.", len(records), skipped
        )
        return records

    def _parse_scene(self, scene_dir: Path) -> Optional[SceneRecord]:
        """Parse one scene directory into a :class:`SceneRecord`.

        Returns ``None`` and logs a warning if the scene is not usable (missing
        RGB image is the only hard requirement; all other fields degrade
        gracefully).

        Args:
            scene_dir: Absolute path to the scene folder.

        Returns:
            Parsed :class:`SceneRecord`, or ``None`` if the scene must be skipped.
        """
        scene_id = str(scene_dir.relative_to(self._sunrgbd_root))

        rgb_relpath = _find_rgb_image(scene_dir, self.raw_root)
        if rgb_relpath is None:
            logger.warning("Skipping scene (no RGB image): %s", scene_id)
            return None

        room_type = _read_room_type(scene_dir)
        objects = _read_objects(scene_dir)
        layout_dims = _read_layout_dims(scene_dir)

        return SceneRecord(
            scene_id=scene_id,
            room_type=room_type,
            objects=objects,
            num_objects=len(objects),
            layout_dims=layout_dims,
            rgb_relpath=rgb_relpath,
        )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(records: list[SceneRecord]) -> None:
    """Print a room-type frequency table to stdout.

    Args:
        records: Parsed scene records.
    """
    counts: Counter[str] = Counter(r.room_type for r in records)
    total = len(records)
    col_w = max((len(k) for k in counts), default=12)

    header = f"{'Room type':<{col_w}}  {'Count':>6}  {'%':>6}"
    separator = "─" * len(header)
    print(f"\n{header}")
    print(separator)
    for room_type, count in counts.most_common():
        pct = 100.0 * count / total if total else 0.0
        print(f"{room_type:<{col_w}}  {count:>6}  {pct:>5.1f}%")
    print(separator)
    print(f"{'TOTAL':<{col_w}}  {total:>6}\n")


def write_json(records: list[SceneRecord], out_path: Path) -> None:
    """Serialise *records* to *out_path* as a pretty-printed UTF-8 JSON array.

    Parent directories are created automatically if they do not exist.

    Args:
        records: Parsed scene records to write.
        out_path: Destination file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in records]
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %d records to %s", len(payload), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse SUN RGB-D scenes into structured_scenes.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Path to the raw data root containing the SUNRGBD/ sub-directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/structured_scenes.json"),
        help="Destination JSON file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N scenes (useful for fast iteration).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    _parser = DataParser(raw_root=args.raw_root, limit=args.limit)
    _records = _parser.load()

    _print_summary(_records)
    write_json(_records, args.out)
