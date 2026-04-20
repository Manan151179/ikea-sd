"""
generator.py — Stable Diffusion 1.5 + ControlNet MLSD image generation.

Two pipelines share the SD 1.5 backbone weights (UNet / VAE / text encoder)
to stay within T4's 15 GB VRAM budget:

    Uncontrolled  StableDiffusionPipeline
    Controlled    StableDiffusionControlNetPipeline (MLSD line map conditioning)

Both pipelines are lazy-loaded on first use and cached for the lifetime of
the Generator object.  Call `unload()` to free VRAM between stages.

Typical memory usage at 512x512 fp16
    Model load   ~5-6 GB
    Generation   ~3 GB peak activation
    Total        ~8-9 GB  (fits in 15 GB with margin)

Public API
----------
Generator(config)
    .generate_uncontrolled(positive, negative, seed) -> PIL.Image
    .generate_controlled(positive, negative, seed, reference_image) -> PIL.Image
    .unload()

run_experiment(scenes, seeds, db_path, out_dir, config, strategies, control_modes)
    Full factorial driver: (scene x strategy x control_mode x seed).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model ID fallback chain
# SD v1.5 moved repos; try the canonical id first, fall back to the mirror.
# ---------------------------------------------------------------------------

_SD15_PRIMARY   = "runwayml/stable-diffusion-v1-5"
_SD15_FALLBACK  = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_MLSD_ANNOTATOR = "lllyasviel/Annotators"

# DPMSolverMultistepScheduler at 25 steps matches DDIM quality at 50.
_RECOMMENDED_STEPS = 25

# CLIP tokenizer hard limit; warn when the prompt will be silently truncated.
_MAX_TOKENS = 77

# HTTP status codes that indicate an auth / gating problem on HuggingFace.
_HF_AUTH_MARKERS = frozenset({"401", "403", "404", "gated", "access"})


# ---------------------------------------------------------------------------
# Token-length guard
# ---------------------------------------------------------------------------

def _warn_if_truncated(prompt: str, tokenizer: object) -> None:
    """Log a warning when *prompt* exceeds the CLIP 77-token limit.

    The diffusers pipeline silently truncates; this surfaces the truncation
    so the researcher can shorten the prompt if quality is affected.

    Args:
        prompt: The text prompt to check.
        tokenizer: A HuggingFace tokenizer whose ``__call__`` returns an
            object with an ``input_ids`` attribute.
    """
    ids = tokenizer(prompt, truncation=False).input_ids
    if len(ids) > _MAX_TOKENS:
        logger.warning(
            "Prompt truncated: %d tokens > limit %d. "
            "Excess tokens are ignored by CLIP. Prompt prefix: %.80s",
            len(ids), _MAX_TOKENS, prompt,
        )


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------

def _load_scheduler(pipe: object) -> None:
    """Replace *pipe*'s scheduler in-place with DPMSolverMultistepScheduler.

    Args:
        pipe: A loaded diffusers pipeline.
    """
    from diffusers import DPMSolverMultistepScheduler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )


def _apply_memory_optimisations(pipe: object) -> None:
    """Enable attention slicing and VAE slicing on *pipe*.

    These are the safe, xformers-free alternatives for T4 Colab images.
    Attention slicing reduces peak VRAM during self-attention at the cost
    of a small throughput penalty (~5-10%).

    Args:
        pipe: A loaded diffusers pipeline.
    """
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()


def _is_auth_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a HuggingFace auth / gating error."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _HF_AUTH_MARKERS)


# ---------------------------------------------------------------------------
# Pipeline loaders
# ---------------------------------------------------------------------------

def _load_uncontrolled_pipeline(
    base_model_id: str,
    torch_dtype: object,
    device: str,
) -> object:
    """Load StableDiffusionPipeline, retrying with the mirror repo on auth errors.

    Args:
        base_model_id: Primary HuggingFace model ID.
        torch_dtype: ``torch.float16``.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Configured, device-resident :class:`~diffusers.StableDiffusionPipeline`.

    Raises:
        RuntimeError: If both the primary and fallback IDs fail to load.
    """
    from diffusers import StableDiffusionPipeline

    candidates = [base_model_id]
    if base_model_id == _SD15_PRIMARY:
        candidates.append(_SD15_FALLBACK)

    last_exc: Optional[Exception] = None
    for model_id in candidates:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            logger.info("Uncontrolled pipeline loaded from: %s", model_id)
            _load_scheduler(pipe)
            _apply_memory_optimisations(pipe)
            pipe.to(device)
            return pipe
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_auth_error(exc):
                logger.warning(
                    "Auth/access error loading %s: %s — trying next candidate.",
                    model_id, exc,
                )
            else:
                raise

    raise RuntimeError(
        f"All model ID candidates failed for uncontrolled pipeline. "
        f"Last error: {last_exc}"
    )


def _load_controlled_pipeline(
    base_model_id: str,
    controlnet_model_id: str,
    torch_dtype: object,
    device: str,
) -> object:
    """Load StableDiffusionControlNetPipeline with MLSD ControlNet weights.

    The ControlNet adapter weights are downloaded first (~330 MB), then the
    SD backbone is loaded (re-used from HF cache if already present).

    Args:
        base_model_id: Primary HuggingFace model ID for the SD backbone.
        controlnet_model_id: HuggingFace ID for the ControlNet MLSD weights.
        torch_dtype: ``torch.float16``.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Configured, device-resident
        :class:`~diffusers.StableDiffusionControlNetPipeline`.

    Raises:
        RuntimeError: If both the primary and fallback backbone IDs fail.
    """
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    logger.info("Loading ControlNet adapter from: %s", controlnet_model_id)
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=torch_dtype,
    )

    candidates = [base_model_id]
    if base_model_id == _SD15_PRIMARY:
        candidates.append(_SD15_FALLBACK)

    last_exc: Optional[Exception] = None
    for model_id in candidates:
        try:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            logger.info("ControlNet pipeline loaded from: %s", model_id)
            _load_scheduler(pipe)
            _apply_memory_optimisations(pipe)
            pipe.to(device)
            return pipe
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_auth_error(exc):
                logger.warning(
                    "Auth/access error loading %s: %s — trying next candidate.",
                    model_id, exc,
                )
            else:
                raise

    raise RuntimeError(
        f"All model ID candidates failed for ControlNet pipeline. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# MLSD preprocessor (lazy singleton per Generator instance)
# ---------------------------------------------------------------------------

class _MLSDPreprocessor:
    """Lazy wrapper around controlnet_aux.MLSDdetector.

    The MLSD detector is ~50 MB and takes ~1 s to initialise.  It is only
    downloaded and loaded on the first call, so runs using only the
    uncontrolled pipeline pay no cost.
    """

    def __init__(self) -> None:
        self._detector: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._detector is not None:
            return
        from controlnet_aux import MLSDdetector

        logger.info("Loading MLSD detector from %s…", _MLSD_ANNOTATOR)
        self._detector = MLSDdetector.from_pretrained(_MLSD_ANNOTATOR)
        logger.info("MLSD detector ready.")

    def __call__(self, image: Image.Image) -> Image.Image:
        """Run MLSD line detection and return the line-map condition image.

        Args:
            image: 512x512 RGB reference scene photo.

        Returns:
            512x512 RGB MLSD line-map suitable for ControlNet conditioning.
        """
        self._ensure_loaded()
        return self._detector(image)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """Lazy-loading wrapper around the SD 1.5 and ControlNet MLSD pipelines.

    Pipelines are instantiated on the first generate_* call and cached.
    Call :meth:`unload` to move them off-device and reclaim VRAM before
    switching between generation modes or running evaluation.

    Args:
        config: Project configuration dict loaded from ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        import torch

        self._config = config
        self._device: str  = config["model"]["device"]
        self._dtype        = (
            torch.float16 if config["model"]["dtype"] == "fp16" else torch.float32
        )
        self._base_model_id: str       = config["model"]["base_model_id"]
        self._controlnet_model_id: str = config["model"]["controlnet_model_id"]

        gen_cfg = config["generation"]
        self._num_steps: int  = gen_cfg.get("num_inference_steps", _RECOMMENDED_STEPS)
        self._guidance: float = gen_cfg.get("guidance_scale", 7.5)
        self._cn_scale: float = gen_cfg.get("controlnet_conditioning_scale", 1.0)
        self._height: int     = gen_cfg.get("height", 512)
        self._width: int      = gen_cfg.get("width", 512)

        self._pipe_uc: Optional[object] = None
        self._pipe_cn: Optional[object] = None
        self._mlsd = _MLSDPreprocessor()

    # ------------------------------------------------------------------
    # Private accessors — each loads its pipeline on first call
    # ------------------------------------------------------------------

    def _get_uncontrolled(self) -> object:
        if self._pipe_uc is None:
            self._pipe_uc = _load_uncontrolled_pipeline(
                self._base_model_id, self._dtype, self._device
            )
        return self._pipe_uc

    def _get_controlled(self) -> object:
        if self._pipe_cn is None:
            self._pipe_cn = _load_controlled_pipeline(
                self._base_model_id,
                self._controlnet_model_id,
                self._dtype,
                self._device,
            )
        return self._pipe_cn

    def _make_generator(self, seed: int) -> object:
        """Return a torch.Generator seeded with *seed*, on CUDA when available."""
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.Generator(device=dev).manual_seed(seed)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_uncontrolled(
        self,
        positive: str,
        negative: Optional[str],
        seed: int,
    ) -> Image.Image:
        """Generate one image from text alone (no reference scene conditioning).

        Args:
            positive: Positive text prompt.
            negative: Optional negative prompt.
            seed: Reproducibility seed.

        Returns:
            :data:`height` x :data:`width` RGB :class:`~PIL.Image.Image`.
        """
        pipe = self._get_uncontrolled()
        _warn_if_truncated(positive, pipe.tokenizer)

        output = pipe(
            prompt=positive,
            negative_prompt=negative,
            height=self._height,
            width=self._width,
            num_inference_steps=self._num_steps,
            guidance_scale=self._guidance,
            generator=self._make_generator(seed),
            num_images_per_prompt=1,
        )
        return output.images[0]

    def generate_controlled(
        self,
        positive: str,
        negative: Optional[str],
        seed: int,
        reference_image: Image.Image,
    ) -> Image.Image:
        """Generate one image conditioned on an MLSD line map from *reference_image*.

        The MLSD detector is applied internally; callers pass the raw RGB
        reference photo and receive a generated image back.

        Args:
            positive: Positive text prompt.
            negative: Optional negative prompt.
            seed: Reproducibility seed.
            reference_image: 512x512 RGB photo of the reference scene.

        Returns:
            :data:`height` x :data:`width` RGB :class:`~PIL.Image.Image`.
        """
        pipe = self._get_controlled()
        _warn_if_truncated(positive, pipe.tokenizer)

        control_image = self._mlsd(reference_image)

        output = pipe(
            prompt=positive,
            negative_prompt=negative,
            image=control_image,
            height=self._height,
            width=self._width,
            num_inference_steps=self._num_steps,
            guidance_scale=self._guidance,
            controlnet_conditioning_scale=self._cn_scale,
            generator=self._make_generator(seed),
            num_images_per_prompt=1,
        )
        return output.images[0]

    # ------------------------------------------------------------------
    # VRAM management
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Move both pipelines to CPU and release VRAM.

        Pipelines are re-initialised from the HF cache on the next
        generate_* call.  Call this before switching between uncontrolled
        and controlled generation if VRAM is tight.
        """
        import torch

        for attr in ("_pipe_uc", "_pipe_cn"):
            pipe = getattr(self, attr)
            if pipe is not None:
                pipe.to("cpu")
                setattr(self, attr, None)
                logger.debug("Unloaded %s.", attr)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                "VRAM after unload: %.1f GB free / %.1f GB total.",
                torch.cuda.memory_reserved() / 1e9,
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
        else:
            logger.info("Pipelines unloaded (no CUDA device).")


# ---------------------------------------------------------------------------
# Output filename helper
# ---------------------------------------------------------------------------

def _make_image_path(
    out_dir: Path,
    strategy: str,
    control_mode: str,
    scene_id: str,
    seed: int,
) -> Path:
    """Return a deterministic, flat PNG path for one generation run.

    Forward/back-slashes in *scene_id* are replaced with ``__`` so the
    filename stays flat inside *out_dir*.

    Args:
        out_dir: Output directory.
        strategy: ``"A"``, ``"B"``, ``"C"``, or ``"D"``.
        control_mode: ``"none"`` or ``"mlsd"``.
        scene_id: Relative scene path from the dataset root.
        seed: Generation seed.

    Returns:
        Full :class:`~pathlib.Path` to the (not-yet-written) PNG.
    """
    safe_scene = re.sub(r"[/\\]", "__", scene_id)
    fname = f"{strategy}_{control_mode}_{safe_scene}_seed{seed}.png"
    return out_dir / fname


# ---------------------------------------------------------------------------
# Batch experiment driver
# ---------------------------------------------------------------------------

def run_experiment(
    scenes: list[dict],
    seeds: list[int],
    db_path: str,
    out_dir: str,
    config: dict,
    strategies: Sequence[str] = ("A", "B", "C", "D"),
    control_modes: Sequence[str] = ("none", "mlsd"),
) -> None:
    """Run the full (scene x strategy x control_mode x seed) factorial experiment.

    Design choices:
    - Already-generated images are skipped (resume-safe).
    - OOM errors flush the CUDA cache and continue rather than aborting.
    - All other exceptions are logged at ERROR level and skipped.
    - DB insertion happens only after a successful save, so the DB stays
      consistent with the filesystem.

    Args:
        scenes: List of scene dicts from ``structured_scenes.json``.
        seeds: Integer seeds to iterate over.
        db_path: Path to the SQLite database (initialised if absent).
        out_dir: Directory for generated PNGs.
        config: Project config dict.
        strategies: Prompt strategies to run (subset of A/B/C/D).
        control_modes: Control modes to run (subset of none/mlsd).
    """
    import torch
    from src import db as _db
    from src.prompt_builder import build_prompts
    from src.utils import load_image

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_root = Path(config["paths"]["data_raw"])

    conn: sqlite3.Connection = _db.init_db(db_path)
    generator = Generator(config)

    total = len(scenes) * len(seeds) * len(strategies) * len(control_modes)
    logger.info(
        "Experiment: %d scenes × %d seeds × %d strategies × %d modes = %d runs.",
        len(scenes), len(seeds), len(strategies), len(control_modes), total,
    )

    with tqdm(total=total, desc="Generating", unit="img") as pbar:
        for scene in scenes:
            scene_id: str = scene["scene_id"]

            for seed in seeds:
                prompts = build_prompts(scene, seed=seed)

                for strategy in strategies:
                    entry       = prompts[strategy]
                    positive    = entry["positive"]
                    negative    = entry["negative"]

                    for control_mode in control_modes:
                        img_path = _make_image_path(
                            out_path, strategy, control_mode, scene_id, seed
                        )

                        # Resume: skip already-completed images
                        if img_path.is_file():
                            logger.debug("Already exists, skipping: %s", img_path.name)
                            pbar.update(1)
                            continue

                        ref_path: Optional[Path] = None
                        if control_mode == "mlsd":
                            ref_path = raw_root / scene["rgb_relpath"]
                            if not ref_path.is_file():
                                logger.warning(
                                    "Reference image missing for mlsd — skipping "
                                    "%s / strategy %s / seed %d: %s",
                                    scene_id, strategy, seed, ref_path,
                                )
                                pbar.update(1)
                                continue

                        try:
                            if control_mode == "mlsd":
                                ref_img = load_image(ref_path)
                                image = generator.generate_controlled(
                                    positive, negative, seed, ref_img
                                )
                            else:
                                image = generator.generate_uncontrolled(
                                    positive, negative, seed
                                )

                            image.save(img_path)

                            _db.insert_run(
                                conn,
                                scene_id=scene_id,
                                room_type=scene["room_type"],
                                strategy=strategy,
                                control_mode=control_mode,
                                seed=seed,
                                positive_prompt=positive,
                                negative_prompt=negative,
                                image_path=str(img_path),
                                reference_image_path=(
                                    str(ref_path) if ref_path is not None else None
                                ),
                            )

                        except RuntimeError as exc:
                            if "out of memory" in str(exc).lower():
                                logger.error(
                                    "OOM: %s / %s / %s / seed%d — "
                                    "clearing CUDA cache and continuing.",
                                    scene_id, strategy, control_mode, seed,
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()
                            else:
                                logger.error(
                                    "RuntimeError: %s / %s / %s / seed%d — %s",
                                    scene_id, strategy, control_mode, seed, exc,
                                )

                        except Exception as exc:  # noqa: BLE001
                            logger.error(
                                "Unexpected error: %s / %s / %s / seed%d — %s",
                                scene_id, strategy, control_mode, seed, exc,
                            )

                        pbar.update(1)

    conn.close()
    logger.info(
        "Experiment complete. Images: %s  Database: %s", out_dir, db_path
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SD generation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenes-json",
        type=Path,
        default=Path("data/processed/structured_scenes.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/generated_images"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of scenes.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
    )
    parser.add_argument(
        "--control-modes",
        nargs="+",
        default=["none", "mlsd"],
        choices=["none", "mlsd"],
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

    scenes: list[dict] = json.loads(
        args.scenes_json.read_text(encoding="utf-8")
    )
    if args.limit is not None:
        scenes = scenes[: args.limit]

    run_experiment(
        scenes=scenes,
        seeds=cfg["generation"]["seeds"],
        db_path=cfg["paths"]["db_path"],
        out_dir=str(args.out_dir),
        config=cfg,
        strategies=args.strategies,
        control_modes=args.control_modes,
    )
