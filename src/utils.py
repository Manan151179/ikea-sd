"""
utils.py — Shared helpers: seeding, image I/O, visualisation, timing, data download.

Public API
----------
set_seed(seed)                   Seed Python / NumPy / PyTorch; return a CUDA Generator.
load_image(path)                 Load any image as a 512×512 RGB PIL.Image.
make_grid(images, labels, ...)   Compose a labelled image grid with Matplotlib.
timer_context(label)             Context manager that logs elapsed wall-clock time.
download_sunrgbd_sample(...)     Fetch the SUN RGB-D metadata MAT; guide user for images.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = 512  # canonical spatial dimension for this project

_SUNRGBD_BASE_URL = "https://rgbd.cs.princeton.edu/data"
_METADATA_FILENAME = "SUNRGBDMeta2DBB_v2.mat"
_METADATA_URL = f"{_SUNRGBD_BASE_URL}/{_METADATA_FILENAME}"

# Friendly instructions printed when automated download of images is impossible.
_MANUAL_DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║           SUN RGB-D — Manual Download Required                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  The SUN RGB-D site does not expose individual scene images;        ║
║  only the full 6.4 GB archive is available for download.            ║
║                                                                      ║
║  To obtain images, do ONE of the following:                         ║
║                                                                      ║
║  Option A — Full dataset (recommended)                              ║
║    wget https://rgbd.cs.princeton.edu/data/SUNRGBD.zip              ║
║    unzip SUNRGBD.zip -d <out_dir>                                   ║
║                                                                      ║
║  Option B — Colab one-liner                                         ║
║    !wget -q https://rgbd.cs.princeton.edu/data/SUNRGBD.zip -P /tmp  ║
║    !unzip -q /tmp/SUNRGBD.zip -d {out_dir}                          ║
║                                                                      ║
║  After extraction, place the SUNRGBD/ folder so that the path       ║
║  <out_dir>/SUNRGBD/kv1/... exists, then re-run data_parser.py.     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# Transient HTTP status codes worth retrying
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 1.0


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> "torch.Generator":  # noqa: F821  (torch imported lazily)
    """Seed Python, NumPy, PyTorch CPU and CUDA, and return a CUDA Generator.

    Calling this before pipeline construction guarantees reproducible image
    outputs for a given seed value.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A :class:`torch.Generator` pinned to CUDA and initialised with *seed*.
        Falls back to a CPU generator when CUDA is not available.
    """
    import torch  # deferred — not available in every environment

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        logger.warning("CUDA not available — returning CPU Generator.")
        generator = torch.Generator(device="cpu").manual_seed(seed)

    logger.debug("Global seed set to %d.", seed)
    return generator


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


def _center_crop_square(img: Image.Image) -> Image.Image:
    """Return the largest centre-aligned square crop of *img*.

    Args:
        img: Source PIL image of any mode.

    Returns:
        Square-cropped PIL image (same mode as input).
    """
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_image(path: str | Path) -> Image.Image:
    """Load an image from *path*, centre-crop to square, resize to 512×512, RGB.

    The centre-crop preserves the compositional centre of the scene (where
    the main furniture is usually located) without distorting aspect ratios.

    Args:
        path: File-system path to any image format supported by Pillow.

    Returns:
        512×512 RGB :class:`~PIL.Image.Image`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        OSError: If Pillow cannot decode the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    img = img.convert("RGB")
    img = _center_crop_square(img)
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# make_grid
# ---------------------------------------------------------------------------


def make_grid(
    images: list[Image.Image],
    labels: list[str],
    rows: int,
    cols: int,
    title: str = "",
    cell_size_inches: float = 3.0,
) -> Image.Image:
    """Compose a labelled image grid and return it as a PIL Image.

    Matplotlib is used for layout so labels can be rendered cleanly below
    each cell.  The figure is rendered in-memory (no file written here).

    Args:
        images: Flat list of PIL images.  Must have exactly ``rows * cols``
            elements.
        labels: Per-image caption strings (same length as *images*).
        rows: Number of grid rows.
        cols: Number of grid columns.
        title: Optional super-title rendered above the grid.
        cell_size_inches: Width/height of each cell in the figure (inches).

    Returns:
        Rendered grid as a PIL RGB Image.

    Raises:
        ValueError: If ``len(images) != rows * cols`` or ``len(labels) != len(images)``.
    """
    import io
    import matplotlib.pyplot as plt  # deferred — large import

    expected = rows * cols
    if len(images) != expected:
        raise ValueError(
            f"Expected {expected} images for a {rows}×{cols} grid, "
            f"got {len(images)}."
        )
    if len(labels) != len(images):
        raise ValueError(
            f"len(labels)={len(labels)} must equal len(images)={len(images)}."
        )

    fig_w = cols * cell_size_inches
    fig_h = rows * cell_size_inches + (0.5 if title else 0.0)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # Normalise axes to a flat list regardless of rows/cols dimensions
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for ax, img, label in zip(axes_flat, images, labels):
        ax.imshow(np.array(img))
        ax.set_title(label, fontsize=7, pad=3, wrap=True)
        ax.axis("off")

    # Hide any unused axes (when len(images) < rows * cols)
    for ax in axes_flat[len(images):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.01)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert("RGB")  # strip alpha; detach from BytesIO


# ---------------------------------------------------------------------------
# timer_context
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def timer_context(label: str = "block") -> Generator[None, None, None]:
    """Context manager that logs wall-clock elapsed time at INFO level.

    Usage::

        with timer_context("SD inference"):
            images = pipe(...)

    Logs::

        INFO  utils  SD inference completed in 4.32 s

    Args:
        label: Human-readable name for the timed block.

    Yields:
        Nothing; the context body runs between enter and exit.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2f s", label, elapsed)


# ---------------------------------------------------------------------------
# download_sunrgbd_sample — internal HTTP helpers
# ---------------------------------------------------------------------------


def _http_get_with_retry(url: str, timeout_s: int = 30) -> bytes:
    """Download *url* with exponential-backoff retry on transient errors.

    Args:
        url: Target URL.
        timeout_s: Per-request timeout in seconds.

    Returns:
        Raw response body bytes.

    Raises:
        RuntimeError: After :data:`_MAX_RETRIES` failed attempts.
        ValueError: On a non-retryable 4xx client error.
    """
    import urllib.error
    import urllib.request

    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_STATUS_CODES:
                raise ValueError(
                    f"Non-retryable HTTP {exc.code} fetching {url}"
                ) from exc
            last_exc = exc
        except urllib.error.URLError as exc:
            last_exc = exc

        delay = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
        logger.warning(
            "Attempt %d/%d failed for %s (%s). Retrying in %.1fs.",
            attempt, _MAX_RETRIES, url, last_exc, delay,
        )
        time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {url} after {_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


def _download_metadata(out_dir: Path) -> Optional[Path]:
    """Download SUNRGBDMeta2DBB_v2.mat into *out_dir* if not already present.

    Args:
        out_dir: Destination directory.

    Returns:
        Path to the downloaded file, or ``None`` on failure.
    """
    dest = out_dir / _METADATA_FILENAME
    if dest.is_file():
        logger.info("Metadata already present: %s", dest)
        return dest

    logger.info("Downloading %s (~4.3 MB) ...", _METADATA_URL)
    try:
        data = _http_get_with_retry(_METADATA_URL)
        dest.write_bytes(data)
        logger.info("Saved metadata to %s", dest)
        return dest
    except Exception as exc:  # noqa: BLE001
        logger.warning("Metadata download failed: %s", exc)
        return None


def _sunrgbd_tree_exists(out_dir: Path) -> bool:
    """Return True if a pre-extracted SUNRGBD/ folder is present under *out_dir*."""
    sunrgbd_path = out_dir / "SUNRGBD"
    if not sunrgbd_path.is_dir():
        return False
    # Verify at least one sensor sub-folder exists
    sensor_dirs = ("kv1", "kv2", "realsense", "xtion")
    return any((sunrgbd_path / s).is_dir() for s in sensor_dirs)


# ---------------------------------------------------------------------------
# download_sunrgbd_sample — public function
# ---------------------------------------------------------------------------


def download_sunrgbd_sample(out_dir: str | Path, n_scenes: int = 50) -> None:
    """Prepare SUN RGB-D data under *out_dir* for a sample run of *n_scenes*.

    What this function does
    -----------------------
    1. Creates *out_dir* if it does not exist.
    2. Downloads ``SUNRGBDMeta2DBB_v2.mat`` (~4.3 MB) — the only individually
       addressable file on the SUN RGB-D server.
    3. If a pre-extracted ``SUNRGBD/`` tree already exists under *out_dir*,
       reports how many scene folders were found.
    4. Otherwise prints clear manual-download instructions, because the SUN
       RGB-D site only exposes a single 6.4 GB archive with no per-scene URLs.

    The ``n_scenes`` argument is passed through to :class:`~src.data_parser.DataParser`
    via the caller; this function cannot sub-sample the archive itself.

    Args:
        out_dir: Root directory where data will be stored.  Will be created
            if absent.
        n_scenes: Desired number of scenes (informational — logged only).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("SUN RGB-D target directory: %s  (requesting %d scenes)", out_dir, n_scenes)

    # Step 1: fetch the small metadata file unconditionally
    _download_metadata(out_dir)

    # Step 2: check for a pre-extracted image tree
    if _sunrgbd_tree_exists(out_dir):
        scene_count = sum(
            1
            for root, _dirs, files in os.walk(out_dir / "SUNRGBD")
            if "scene.txt" in files
        )
        logger.info(
            "Found existing SUNRGBD tree with %d scene folders. "
            "Ready to pass --raw-root %s to data_parser.",
            scene_count,
            out_dir,
        )
        return

    # Step 3: no pre-extracted tree — cannot download images automatically
    logger.warning(
        "No extracted SUNRGBD/ folder found under %s. "
        "Individual scene image URLs are not available on the SUN RGB-D server; "
        "only the full 6.4 GB archive can be fetched.",
        out_dir,
    )
    print(_MANUAL_DOWNLOAD_INSTRUCTIONS.format(out_dir=out_dir))
