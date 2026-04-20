"""
evaluator.py — Compute alignment, consistency, diversity, and aesthetic metrics.

Four metrics are computed for every run in the database:

    clip_score   CLIP cosine similarity (image vs positive prompt) x 100.
                 Uses open_clip ViT-B-32.  Range roughly 20-35 for good SD output.

    lpips        Mean pairwise LPIPS over the 3 seeds in a (scene, strategy,
                 control_mode) group.  Lower = more consistent across seeds.
                 Uses lpips library with AlexNet backbone, images at 256x256.

    diversity    Mean pairwise (1 - cosine) of CLIP image embeddings across
                 seeds in the same group.  Higher = more visually diverse.
                 Complementary to lpips: together they show the consistency /
                 diversity trade-off of each prompt strategy.

    aesthetic    LAION aesthetic score (1-10) predicted by a small MLP head
                 on top of the CLIP image embedding.
                 Fallback: if the MLP weights cannot be downloaded, the score
                 is approximated as clip_image_embed.norm(dim=-1) / 40, clamped
                 to [1, 10].  This proxy is documented in _aesthetic_fallback().

Public API
----------
evaluate_all(db_path, config, force=False, out_csv=None) -> pd.DataFrame
    Batch-encode all images and prompts, compute all four metrics, write to
    DB, and return a wide-format DataFrame.  Already-computed metrics are
    skipped unless force=True.
"""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LAION improved aesthetic predictor weights (~4 MB linear MLP)
_AESTHETIC_MLP_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)

# LPIPS and CLIP processing size
_LPIPS_SIZE = 256
_CLIP_MODEL  = "ViT-B-32"
_CLIP_PRETRAINED = "openai"

# Metric names as stored in the DB
METRIC_CLIP_SCORE  = "clip_score"
METRIC_LPIPS       = "lpips"
METRIC_DIVERSITY   = "diversity"
METRIC_AESTHETIC   = "aesthetic"

ALL_METRICS = (METRIC_CLIP_SCORE, METRIC_LPIPS, METRIC_DIVERSITY, METRIC_AESTHETIC)


# ---------------------------------------------------------------------------
# Aesthetic predictor
# ---------------------------------------------------------------------------

class _AestheticMLP(torch.nn.Module):
    """Small linear MLP head used by the LAION improved aesthetic predictor.

    Architecture matches the published checkpoint exactly:
    512 -> 256 -> 128 -> 64 -> 16 -> 1 with ReLU activations and dropout.

    Args:
        input_dim: Dimensionality of the CLIP image embedding (512 for ViT-B-32).
    """

    def __init__(self, input_dim: int = 512) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _load_aesthetic_mlp(device: str) -> Optional[_AestheticMLP]:
    """Download and return the LAION aesthetic MLP head.

    Returns None on any failure so the caller can switch to the fallback proxy.

    Args:
        device: Target device string (``"cuda"`` or ``"cpu"``).

    Returns:
        Loaded :class:`_AestheticMLP` in eval mode, or ``None``.
    """
    try:
        logger.info("Downloading aesthetic predictor weights…")
        raw = urllib.request.urlopen(_AESTHETIC_MLP_URL, timeout=30).read()
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        model = _AestheticMLP(input_dim=512)
        model.load_state_dict(state)
        model.eval().to(device)
        logger.info("Aesthetic MLP loaded.")
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load aesthetic MLP (%s) — using image-statistics fallback.", exc
        )
        return None


def _aesthetic_fallback(images: list[Image.Image]) -> torch.Tensor:
    """Proxy aesthetic score from image sharpness and colorfulness.

    Used when the LAION MLP weights cannot be downloaded.  The L2-norm of
    CLIP embeddings is NOT used here because it concentrates tightly for
    fixed-dimension distributions (chi distribution), giving near-zero
    variance across images and making it useless as a ranking signal.

    Instead, two complementary pixel-space statistics are combined:
    - **Sharpness**: variance of the image Laplacian (blurry or plain images
      score low; crisp, detailed rooms score high).
    - **Colorfulness**: Hasler & Suesstrunk (2003) metric — RMS of the
      per-pixel opponent-channel deviations from the mean.

    Both are normalised to [0, 1] with empirically chosen caps and blended
    60/40 before mapping to the 1–10 scale.

    This is a rough proxy only and should be interpreted as a relative
    ranking within a run, not as an absolute aesthetic score.

    Args:
        images: List of PIL RGB images.

    Returns:
        Float32 tensor of shape (N,) with scores in [1, 10].
    """
    scores: list[float] = []
    for img in images:
        # Downsample to 64x64 for speed; enough for global statistics
        arr = np.array(img.resize((64, 64)).convert("RGB"), dtype=np.float32) / 255.0

        # Sharpness: variance of gradient magnitude on luminance channel
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gy, gx = np.gradient(gray)
        sharpness = float(np.var(gx) + np.var(gy))
        sharpness = min(sharpness / 0.02, 1.0)   # cap: 0.02 ≈ 95th pct for SD output

        # Colorfulness: Hasler & Suesstrunk (2003)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorfulness = float(
            np.sqrt(rg.std() ** 2 + yb.std() ** 2)
            + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        )
        colorfulness = min(colorfulness / 0.3, 1.0)   # cap: 0.3 ≈ typical interior

        score = 1.0 + 9.0 * (0.6 * sharpness + 0.4 * colorfulness)
        scores.append(float(np.clip(score, 1.0, 10.0)))

    return torch.tensor(scores, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CLIP encoder (batched)
# ---------------------------------------------------------------------------

class _CLIPEncoder:
    """Wraps open_clip to produce normalised image and text embeddings in batches.

    Args:
        model_name: open_clip model name, e.g. ``"ViT-B-32"``.
        pretrained: Checkpoint name, e.g. ``"openai"``.
        device: Target device.
        batch_size: Images processed per forward pass.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str,
        batch_size: int = 32,
    ) -> None:
        import open_clip

        logger.info("Loading CLIP %s / %s…", model_name, pretrained)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._tokenize = open_clip.get_tokenizer(model_name)
        self._model.eval().to(device)
        self._device = device
        self._batch_size = batch_size

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Return L2-normalised image embeddings, shape (N, D).

        Args:
            images: List of RGB PIL images (any size; will be preprocessed).

        Returns:
            Float32 tensor on CPU, shape (N, D).
        """
        all_embeds: list[torch.Tensor] = []
        for i in range(0, len(images), self._batch_size):
            batch = images[i : i + self._batch_size]
            tensors = torch.stack(
                [self._preprocess(img) for img in batch]
            ).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(tensors).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu())
        return torch.cat(all_embeds, dim=0)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalised text embeddings, shape (N, D).

        Args:
            texts: List of prompt strings.

        Returns:
            Float32 tensor on CPU, shape (N, D).
        """
        all_embeds: list[torch.Tensor] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            tokens = self._tokenize(batch).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().float())
        return torch.cat(all_embeds, dim=0)


# ---------------------------------------------------------------------------
# LPIPS encoder (batched)
# ---------------------------------------------------------------------------

def _load_lpips(net: str, device: str) -> object:
    """Load and return the LPIPS loss network.

    Args:
        net: Backbone name, e.g. ``"alex"``.
        device: Target device.

    Returns:
        LPIPS model in eval mode on *device*.
    """
    import lpips

    model = lpips.LPIPS(net=net, verbose=False)
    model.eval().to(device)
    return model


def _pil_to_lpips_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """Convert a PIL image to an LPIPS-compatible tensor in [-1, 1].

    Resizes to :data:`_LPIPS_SIZE` x :data:`_LPIPS_SIZE`, converts to float32,
    and maps pixel range [0, 255] to [-1, 1].

    Args:
        img: Input PIL image.
        device: Target device.

    Returns:
        Float tensor of shape (1, 3, H, W) on *device*.
    """
    img = img.resize((_LPIPS_SIZE, _LPIPS_SIZE), Image.LANCZOS).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0     # [0,255] -> [-1,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t.to(device)


# ---------------------------------------------------------------------------
# Per-metric computation helpers
# ---------------------------------------------------------------------------

def _compute_clip_scores(
    img_embeds: torch.Tensor,
    txt_embeds: torch.Tensor,
) -> np.ndarray:
    """Compute cosine similarity * 100 for each (image, text) pair.

    Args:
        img_embeds: L2-normalised image embeddings, shape (N, D).
        txt_embeds: L2-normalised text embeddings, shape (N, D).

    Returns:
        Float32 numpy array of shape (N,) with scores in [0, 100].
    """
    scores = (img_embeds * txt_embeds).sum(dim=-1).clamp(min=0.0) * 100.0
    return scores.numpy().astype(np.float32)


def _compute_aesthetic_scores(
    img_embeds: torch.Tensor,
    mlp: Optional[_AestheticMLP],
    device: str,
    images: Optional[list[Image.Image]] = None,
) -> np.ndarray:
    """Compute aesthetic scores for all images using the MLP or image-stat fallback.

    Args:
        img_embeds: L2-normalised image embeddings, shape (N, D).
            Used as MLP input when *mlp* is provided.
        mlp: Loaded :class:`_AestheticMLP`, or ``None`` to use the fallback.
        device: Device on which to run the MLP.
        images: Original PIL images, required when *mlp* is ``None``.
            Ignored when *mlp* is provided.

    Returns:
        Float32 numpy array of shape (N,) with scores in [1, 10].

    Raises:
        ValueError: If *mlp* is ``None`` and *images* is not supplied.
    """
    if mlp is not None:
        with torch.no_grad():
            scores = mlp(img_embeds.to(device)).squeeze(-1).cpu()
        scores = scores.clamp(min=1.0, max=10.0)
    else:
        if images is None:
            raise ValueError(
                "images must be supplied when mlp=None to use the fallback."
            )
        scores = _aesthetic_fallback(images)
    return scores.numpy().astype(np.float32)


def _compute_group_metrics(
    group_df: pd.DataFrame,
    img_embeds: torch.Tensor,
    images: list[Image.Image],
    lpips_model: object,
    device: str,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute pairwise LPIPS and CLIP-diversity for a seed group.

    A "group" is all runs sharing the same (scene_id, strategy, control_mode).
    With 3 seeds there are 3 pairs; the mean over pairs is reported per run.

    Args:
        group_df: Rows from the runs DataFrame belonging to this group.
        img_embeds: All image embeddings aligned to the full runs DataFrame index.
        images: All loaded PIL images aligned to the full runs DataFrame index.
        lpips_model: Loaded LPIPS model.
        device: Device for LPIPS tensors.

    Returns:
        Two dicts mapping run_id -> float:
        - lpips_scores: mean pairwise LPIPS for the group (same for all members).
        - diversity_scores: mean pairwise (1 - cosine) for the group.
    """
    indices = group_df.index.tolist()
    run_ids = group_df["run_id"].tolist()

    if len(indices) < 2:
        # Single-seed group: LPIPS and diversity are undefined; store NaN.
        nan_val = float("nan")
        return (
            {rid: nan_val for rid in run_ids},
            {rid: nan_val for rid in run_ids},
        )

    # All pairs (i, j) with i < j
    pairs = [(a, b) for idx_a, a in enumerate(indices)
             for b in indices[idx_a + 1:]]

    lpips_vals: list[float] = []
    div_vals: list[float] = []

    for pos_a, pos_b in pairs:
        # LPIPS
        ta = _pil_to_lpips_tensor(images[pos_a], device)
        tb = _pil_to_lpips_tensor(images[pos_b], device)
        with torch.no_grad():
            lp = float(lpips_model(ta, tb).item())
        lpips_vals.append(lp)

        # CLIP diversity: 1 - cosine
        ea = img_embeds[pos_a]
        eb = img_embeds[pos_b]
        cos = float((ea * eb).sum().clamp(-1.0, 1.0).item())
        div_vals.append(1.0 - cos)

    mean_lpips = float(np.mean(lpips_vals))
    mean_div   = float(np.mean(div_vals))

    # Every run in the group receives the same group-level values
    return (
        {rid: mean_lpips for rid in run_ids},
        {rid: mean_div   for rid in run_ids},
    )


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------

def _get_already_computed(conn: object, metric: str) -> set[int]:
    """Return the set of run_ids for which *metric* is already in the DB.

    Args:
        conn: Open database connection.
        metric: Metric name string.

    Returns:
        Set of integer run_ids.
    """
    from src.db import query_df

    df = query_df(
        conn,
        "SELECT run_id FROM metrics WHERE metric_name = ?",
        (metric,),
    )
    if df.empty:
        return set()
    return set(df["run_id"].tolist())


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_all(
    db_path: str,
    config: dict,
    force: bool = False,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate all runs in the database and return a wide-format DataFrame.

    Processing order:
    1. Load all runs from the DB.
    2. Determine which (run_id, metric) pairs still need computation
       (skip if already present and force=False).
    3. Load all required images once.
    4. Batch-encode all images and prompts with CLIP.
    5. Compute clip_score and aesthetic per run.
    6. Compute lpips and diversity per (scene, strategy, control_mode) group.
    7. Write metrics to DB and return a merged wide-format DataFrame.

    Args:
        db_path: Path to the SQLite database produced by :mod:`src.db`.
        config: Project config dict (uses ``model.device``,
            ``evaluation.clip_model_name``, ``evaluation.lpips_net``).
        force: Re-compute and overwrite metrics even if they exist in the DB.
        out_csv: If set, the wide-format DataFrame is also saved here.

    Returns:
        Wide-format :class:`~pandas.DataFrame` with one row per run and
        columns: all runs fields + clip_score, lpips, diversity, aesthetic.
    """
    from src.db import init_db, insert_metric, query_df

    device = config["model"]["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA requested but not available — falling back to CPU.")
        device = "cpu"

    clip_model_name: str = config["evaluation"].get("clip_model_name", _CLIP_MODEL)
    lpips_net: str       = config["evaluation"].get("lpips_net", "alex")

    conn = init_db(db_path)

    # ── 1. Load all runs ──────────────────────────────────────────────────
    runs_df = query_df(conn, "SELECT * FROM runs ORDER BY run_id")
    if runs_df.empty:
        logger.warning("No runs found in database. Run the generator first.")
        return runs_df

    logger.info("Evaluating %d runs.", len(runs_df))

    # ── 2. Determine what needs computing ─────────────────────────────────
    if force:
        pending_run_ids = set(runs_df["run_id"].tolist())
    else:
        computed: dict[str, set[int]] = {
            m: _get_already_computed(conn, m) for m in ALL_METRICS
        }
        # A run needs (re)evaluation if ANY metric is missing
        pending_run_ids = set()
        for m in ALL_METRICS:
            pending_run_ids |= set(runs_df["run_id"].tolist()) - computed[m]

    if not pending_run_ids:
        logger.info("All metrics already computed. Use --force to recompute.")
        return _build_wide_df(conn, runs_df, out_csv)

    pending_df = runs_df[runs_df["run_id"].isin(pending_run_ids)].copy()
    logger.info("%d runs need evaluation.", len(pending_df))

    # ── 3. Load images ─────────────────────────────────────────────────────
    logger.info("Loading %d images…", len(pending_df))
    images: list[Optional[Image.Image]] = []
    valid_mask: list[bool] = []

    for _, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc="Loading images"):
        path = Path(row["image_path"])
        if path.is_file():
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_mask.append(True)
            except OSError as exc:
                logger.warning("Cannot open image %s: %s", path, exc)
                images.append(None)
                valid_mask.append(False)
        else:
            logger.warning("Image file missing: %s", path)
            images.append(None)
            valid_mask.append(False)

    # Work only on rows with loadable images
    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    valid_df      = pending_df.iloc[valid_indices].reset_index(drop=True)
    valid_images  = [images[i] for i in valid_indices]

    if valid_df.empty:
        logger.error("No valid images found — cannot compute any metrics.")
        return _build_wide_df(conn, runs_df, out_csv)

    # ── 4. Batch CLIP encoding ─────────────────────────────────────────────
    clip_enc = _CLIPEncoder(
        model_name=clip_model_name,
        pretrained=_CLIP_PRETRAINED,
        device=device,
    )

    logger.info("Encoding %d images with CLIP…", len(valid_images))
    img_embeds = clip_enc.encode_images(valid_images)

    logger.info("Encoding %d prompts with CLIP…", len(valid_df))
    txt_embeds = clip_enc.encode_texts(valid_df["positive_prompt"].tolist())

    # ── 5. Per-run metrics: clip_score and aesthetic ───────────────────────
    clip_scores = _compute_clip_scores(img_embeds, txt_embeds)

    aesthetic_mlp = _load_aesthetic_mlp(device)
    aesthetic_scores = _compute_aesthetic_scores(
        img_embeds, aesthetic_mlp, device, images=valid_images
    )

    # ── 6. Group metrics: lpips and diversity ──────────────────────────────
    logger.info("Loading LPIPS model…")
    lpips_model = _load_lpips(lpips_net, device)

    lpips_map:     dict[int, float] = {}
    diversity_map: dict[int, float] = {}

    groups = valid_df.groupby(["scene_id", "strategy", "control_mode"])
    logger.info("Computing group metrics across %d groups…", len(groups))

    for _, group_df in tqdm(groups, desc="Group metrics", unit="group"):
        lp, dv = _compute_group_metrics(
            group_df, img_embeds, valid_images, lpips_model, device
        )
        lpips_map.update(lp)
        diversity_map.update(dv)

    # ── 7. Write metrics to DB ─────────────────────────────────────────────
    logger.info("Writing metrics to database…")
    for i, row in valid_df.iterrows():
        run_id = int(row["run_id"])
        insert_metric(conn, run_id, METRIC_CLIP_SCORE,  float(clip_scores[i]))
        insert_metric(conn, run_id, METRIC_AESTHETIC,   float(aesthetic_scores[i]))
        if run_id in lpips_map:
            insert_metric(conn, run_id, METRIC_LPIPS,      lpips_map[run_id])
            insert_metric(conn, run_id, METRIC_DIVERSITY,  diversity_map[run_id])

    logger.info("Metrics written.")
    return _build_wide_df(conn, runs_df, out_csv)


# ---------------------------------------------------------------------------
# Wide DataFrame assembly
# ---------------------------------------------------------------------------

def _build_wide_df(
    conn: object,
    runs_df: pd.DataFrame,
    out_csv: Optional[str],
) -> pd.DataFrame:
    """Join runs with their metrics into a wide-format DataFrame.

    Args:
        conn: Open database connection.
        runs_df: Runs DataFrame (all columns from the runs table).
        out_csv: Optional path to write the CSV.

    Returns:
        Wide-format DataFrame with metric columns appended.
    """
    from src.db import query_df

    metrics_df = query_df(
        conn,
        "SELECT run_id, metric_name, value FROM metrics",
    )

    if metrics_df.empty:
        logger.warning("No metrics in DB yet.")
        return runs_df

    wide = metrics_df.pivot(
        index="run_id", columns="metric_name", values="value"
    ).reset_index()

    # Ensure all expected metric columns are present (fill NaN if missing)
    for m in ALL_METRICS:
        if m not in wide.columns:
            wide[m] = float("nan")

    result = runs_df.merge(wide, on="run_id", how="left")

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info("Metrics CSV saved to %s", out_path)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all generated images and populate the metrics table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the SQLite results database.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/metrics.csv"),
        help="Destination for the wide-format metrics CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite already-computed metrics.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


if __name__ == "__main__":
    import yaml

    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    cfg: dict = yaml.safe_load(args.config.read_text())

    result_df = evaluate_all(
        db_path=str(args.db_path),
        config=cfg,
        force=args.force,
        out_csv=str(args.out_csv),
    )
    print(result_df[["run_id", "scene_id", "strategy", "control_mode",
                      "clip_score", "lpips", "diversity", "aesthetic"]].to_string(index=False))
