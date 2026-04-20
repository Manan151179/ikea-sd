"""
prompt_builder.py — Convert a structured scene record into SD prompt variants.

Four strategies are produced for every scene:

    A  NAIVE       "a {room_type}"
    B  TEMPLATE    slot-fill with filtered top-k object labels
    C  ENRICHED    template + style + material + lighting tokens
    D  ENRICHED+   same positive as C plus a fixed negative prompt

All four are returned by :func:`build_prompts` as a dict keyed by strategy
letter.  Each value is::

    {
        "positive":       str,
        "negative":       str | None,
        "strategy_name":  str,
    }

Style, material, and lighting for strategy C/D are sampled deterministically
from curated pools given a (scene_id, seed) pair, so results are reproducible
across runs while varying meaningfully across scenes and seeds.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Filtering: labels that carry no useful visual signal for a room prompt
# ---------------------------------------------------------------------------

STOPWORD_OBJECTS: frozenset[str] = frozenset(
    {
        # structural surfaces
        "wall", "floor", "ceiling", "ground", "roof",
        # transparent / ubiquitous fixtures
        "window", "door", "doorway", "wall_s", "floor_s",
        # generic human presence
        "person", "people", "human", "man", "woman", "child",
        # ultra-generic catch-alls that add no specificity
        "object", "thing", "item", "stuff", "unknown",
        # light sources — captured by the lighting slot instead
        "light", "lighting", "lamp", "bulb",
    }
)

TOP_K_OBJECTS = 5

# ---------------------------------------------------------------------------
# Style pools keyed by SUN RGB-D room type (snake_case)
# ---------------------------------------------------------------------------

# Generic fallback used when room_type is not in the map
_GENERIC_STYLE_POOL: list[str] = [
    "scandinavian",
    "minimalist",
    "modern",
    "transitional",
    "contemporary",
]

ROOM_STYLE_POOLS: dict[str, list[str]] = {
    "living_room": [
        "scandinavian",
        "mid-century modern",
        "minimalist",
        "industrial",
        "bohemian",
    ],
    "bedroom": [
        "scandinavian",
        "japandi",
        "minimalist",
        "coastal",
        "art deco",
    ],
    "kitchen": [
        "modern farmhouse",
        "scandinavian",
        "industrial",
        "minimalist",
        "mediterranean",
    ],
    "bathroom": [
        "spa",
        "minimalist",
        "industrial",
        "coastal",
        "art deco",
    ],
    "dining_room": [
        "mid-century modern",
        "farmhouse",
        "minimalist",
        "industrial",
        "french country",
    ],
    "office": [
        "minimalist",
        "industrial",
        "scandinavian",
        "modern",
        "japandi",
    ],
    "study": [
        "traditional",
        "minimalist",
        "industrial",
        "scandinavian",
        "mid-century modern",
    ],
    "corridor": [
        "minimalist",
        "scandinavian",
        "modern",
        "industrial",
        "transitional",
    ],
    "hallway": [
        "minimalist",
        "scandinavian",
        "modern",
        "transitional",
        "contemporary",
    ],
    "classroom": [
        "modern",
        "minimalist",
        "scandinavian",
        "industrial",
        "contemporary",
    ],
    "lab": [
        "modern",
        "industrial",
        "minimalist",
        "high-tech",
        "contemporary",
    ],
    "conference_room": [
        "corporate modern",
        "minimalist",
        "industrial",
        "scandinavian",
        "contemporary",
    ],
    "reception": [
        "modern",
        "art deco",
        "minimalist",
        "luxury",
        "contemporary",
    ],
    "gym": [
        "industrial",
        "modern",
        "minimalist",
        "high-tech",
        "contemporary",
    ],
    "storage_room": [
        "minimalist",
        "industrial",
        "modern",
        "scandinavian",
        "utilitarian",
    ],
}

# ---------------------------------------------------------------------------
# Material and lighting pools (shared across all room types)
# ---------------------------------------------------------------------------

_MATERIAL_POOL: list[str] = ["wood", "marble", "linen", "leather", "concrete"]

_LIGHTING_POOL: list[str] = [
    "warm natural",
    "soft diffused",
    "golden hour",
    "cool daylight",
]

# ---------------------------------------------------------------------------
# Fixed negative prompt (strategy D)
# ---------------------------------------------------------------------------

NEGATIVE_PROMPT: str = (
    "low quality, blurry, cartoon, distorted, warped perspective, people, "
    "text, watermark, deformed furniture, oversaturated, extra walls, floating objects"
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_room_type(room_type: str) -> str:
    """Return a display-friendly room type string (underscores → spaces).

    Args:
        room_type: Raw snake_case room type from the scene record.

    Returns:
        Human-readable string, e.g. ``"living room"``.
    """
    return room_type.replace("_", " ")


def _select_top_objects(objects: list[str]) -> list[str]:
    """Filter stopwords and return up to :data:`TOP_K_OBJECTS` labels.

    Preserves the original ordering (already alphabetically sorted by the
    parser) after removing stopwords, then truncates to TOP_K.

    Args:
        objects: Full, deduplicated object label list from the scene record.

    Returns:
        Filtered list of at most :data:`TOP_K_OBJECTS` strings.
    """
    filtered = [obj for obj in objects if obj not in STOPWORD_OBJECTS]
    return filtered[:TOP_K_OBJECTS]


def _oxford_join(items: list[str]) -> str:
    """Join a list of strings with Oxford-comma style.

    Examples::

        []          -> ""
        ["sofa"]    -> "sofa"
        ["a", "b"]  -> "a and b"
        ["a","b","c"] -> "a, b, and c"

    Args:
        items: Strings to join.

    Returns:
        A single joined string.
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _seeded_choice(pool: list[str], rng: random.Random) -> str:
    """Return a deterministic random choice from *pool* using *rng*.

    Args:
        pool: Non-empty list of candidates.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        One element from *pool*.
    """
    return rng.choice(pool)


def _make_rng(scene_id: str, seed: int) -> random.Random:
    """Create a :class:`random.Random` seeded on ``hash(scene_id) ^ seed``.

    Hashing the scene_id ensures different scenes get different style samples
    even at the same seed value.

    Args:
        scene_id: Unique scene identifier string.
        seed: Caller-supplied integer seed.

    Returns:
        Seeded :class:`random.Random` instance.
    """
    combined = hash(scene_id) ^ seed
    return random.Random(combined)


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------


def _build_naive(room_type: str) -> str:
    """Strategy A: minimal single-phrase prompt.

    Args:
        room_type: Snake_case room type.

    Returns:
        Prompt string.
    """
    return f"a {_normalise_room_type(room_type)}"


def _build_template(room_type: str, objects: list[str]) -> str:
    """Strategy B: slot-filled template with filtered object list.

    Args:
        room_type: Snake_case room type.
        objects: Full object label list from the scene record.

    Returns:
        Prompt string, degrading gracefully when no informative objects exist.
    """
    top_objects = _select_top_objects(objects)
    display_room = _normalise_room_type(room_type)

    if top_objects:
        object_phrase = _oxford_join(top_objects)
        return (
            f"a photo of a {display_room}, "
            f"containing {object_phrase}, "
            f"realistic interior"
        )
    return f"a photo of a {display_room}, realistic interior"


def _build_enriched(
    room_type: str,
    objects: list[str],
    rng: random.Random,
) -> str:
    """Strategy C: template augmented with style, material, and lighting tokens.

    Style is sampled from :data:`ROOM_STYLE_POOLS` for the given room type
    (falling back to the generic pool when unrecognised).  Material and
    lighting are sampled from their respective shared pools.

    Args:
        room_type: Snake_case room type.
        objects: Full object label list from the scene record.
        rng: Seeded RNG instance for deterministic sampling.

    Returns:
        Enriched positive prompt string.
    """
    style_pool = ROOM_STYLE_POOLS.get(room_type, _GENERIC_STYLE_POOL)
    style = _seeded_choice(style_pool, rng)
    material = _seeded_choice(_MATERIAL_POOL, rng)
    lighting = _seeded_choice(_LIGHTING_POOL, rng)

    top_objects = _select_top_objects(objects)
    display_room = _normalise_room_type(room_type)

    parts = [f"a photo of a {style} {display_room}"]
    if top_objects:
        parts.append(f"containing {_oxford_join(top_objects)}")
    parts += [
        f"{material} surfaces",
        f"{lighting} lighting",
        "interior design photography",
        "wide angle",
        "4k",
        "architectural digest style",
    ]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompts(scene: dict, seed: int) -> dict[str, dict]:
    """Generate all four prompt variants for a single scene record.

    Args:
        scene: A scene record dict as produced by :mod:`src.data_parser`.
            Must contain at least ``"scene_id"``, ``"room_type"``, and
            ``"objects"`` keys.
        seed: Integer seed controlling deterministic style/material/lighting
            sampling for strategies C and D.

    Returns:
        Dict with keys ``"A"``, ``"B"``, ``"C"``, ``"D"``.  Each value is::

            {
                "positive":      str,
                "negative":      str | None,
                "strategy_name": str,
            }

    Raises:
        KeyError: If *scene* is missing a required field.
    """
    scene_id: str = scene["scene_id"]
    room_type: str = scene["room_type"]
    objects: list[str] = scene["objects"]

    rng = _make_rng(scene_id, seed)

    positive_a = _build_naive(room_type)
    positive_b = _build_template(room_type, objects)
    positive_c = _build_enriched(room_type, objects, rng)

    return {
        "A": {
            "positive": positive_a,
            "negative": None,
            "strategy_name": "naive",
        },
        "B": {
            "positive": positive_b,
            "negative": None,
            "strategy_name": "template",
        },
        "C": {
            "positive": positive_c,
            "negative": None,
            "strategy_name": "enriched",
        },
        "D": {
            "positive": positive_c,
            "negative": NEGATIVE_PROMPT,
            "strategy_name": "enriched_negative",
        },
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


def _demo(processed_json: Path, seed: int) -> None:
    """Load the first 3 scenes from *processed_json* and print all prompts.

    Args:
        processed_json: Path to ``data/processed/structured_scenes.json``.
        seed: Seed passed to :func:`build_prompts`.
    """
    if not processed_json.is_file():
        print(f"File not found: {processed_json}", file=sys.stderr)
        sys.exit(1)

    scenes: list[dict] = json.loads(processed_json.read_text(encoding="utf-8"))
    sample = scenes[:3]

    for scene in sample:
        print(f"\n{'═' * 70}")
        print(f"  scene_id  : {scene['scene_id']}")
        print(f"  room_type : {scene['room_type']}")
        print(f"  objects   : {scene['objects'][:8]}")
        print(f"{'─' * 70}")

        prompts = build_prompts(scene, seed=seed)
        for strategy_key in ("A", "B", "C", "D"):
            entry = prompts[strategy_key]
            label = f"[{strategy_key}] {entry['strategy_name'].upper()}"
            print(f"\n  {label}")
            print(f"    + {entry['positive']}")
            if entry["negative"]:
                print(f"    - {entry['negative']}")

    print(f"\n{'═' * 70}\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo: print prompt variants for the first 3 scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenes-json",
        type=Path,
        default=Path("data/processed/structured_scenes.json"),
        help="Path to structured_scenes.json produced by data_parser.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for style/material/lighting sampling.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    _demo(args.scenes_json, args.seed)
