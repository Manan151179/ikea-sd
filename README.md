# IKEA Interior Design Generation with Stable Diffusion

## Overview

Generates photorealistic interior room images conditioned on IKEA scene data using
Stable Diffusion v1.5 + ControlNet (MLSD line detector). Given a structured scene
description (room type, furniture list, style), the system builds a prompt, runs
controlled generation, and evaluates the outputs with CLIP, LPIPS, and FID.

## Setup

### Colab (recommended — T4 GPU)

```bash
# 1. Mount Drive and clone
from google.colab import drive
drive.mount('/content/drive')
!git clone <repo-url> /content/ikea-sd && cd /content/ikea-sd

# 2. Install dependencies (torch is already present on Colab — skip it)
!pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 \
    $(grep -v '^torch' requirements.txt | grep -v '^#' | grep -v '^$')
```

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
import yaml
from src.data_parser import DataParser
from src.prompt_builder import PromptBuilder
from src.generator import Generator
from src.evaluator import Evaluator
from src.db import ResultsDB

config = yaml.safe_load(open("config.yaml"))

scenes   = DataParser(config).load()
prompts  = [PromptBuilder(config).build(s) for s in scenes]
images   = Generator(config).run(prompts)
metrics  = Evaluator(config).evaluate(images, prompts)
ResultsDB(config).save(metrics)
```

## Architecture

The system is composed of six modules:

| Layer | Module | Responsibility |
|-------|--------|---------------|
| 1 | `data_parser` | Parse raw IKEA `.mat` / JSON scene files into typed dataclasses |
| 2 | `prompt_builder` | Convert structured scene descriptions into SD prompt strings |
| 3 | `generator` | Run ControlNet-guided SD inference; write images to disk |
| 4 | `evaluator` | Compute CLIP similarity, LPIPS, and FID against reference crops |
| 5 | `db` | Persist per-image metadata and metrics to SQLite |
| 6 | `utils` | Shared helpers: config loading, logging, seed management |

All configuration lives in `config.yaml`; no magic values appear in source code.

## Results

_Placeholder — populate after experiments._

| Metric | Value |
|--------|-------|
| CLIP similarity (mean) | — |
| LPIPS (mean) | — |
| FID | — |
