# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Run the pipeline
uv run modules/cli.py --config configs/<config_file>.yaml

# Run with OmegaConf CLI overrides
uv run modules/cli.py --config configs/config_isaacsim.yaml data.0.inputs.rgb=/path/to/video.mp4

# Run tests
uv run pytest tests/
uv run pytest tests/schema/test_schema.py -v          # schema validation tests only
uv run pytest tests/schema/test_schema.py::TestValidConfigs::test_minimal_cosmos_transfer -v  # single test

# Lint
uv run ruff check modules/
uv run ruff format --check modules/

# Docker inference (GPU required)
sudo docker run -it --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  --env-file examples/.env \
  -v "$(pwd)/modules:/app/modules" \
  -v "$(pwd)/configs:/app/configs" \
  paidf-augmentation:latest \
  uv run modules/cli.py --config configs/<config_file>.yaml
```

## Architecture

The pipeline processes media through: **Captioning -> Generation -> Evaluation**, orchestrated by `modules/cli.py`.

### Unified Pydantic Schema (`modules/aug_utils/schema/`)

All YAML configs are validated against a single `PipelineConfig` root model. The schema has 6 top-level sections:

| Section | Purpose |
|---------|---------|
| `data` | List of input/output sample paths (rgb, controls, video, caption, metadata) |
| `endpoints` | API endpoint URLs/models for vlm, llm, cosmos_transfer, cosmos_predict, image_edit |
| `pipeline` | Retry count, logging level, evaluation settings (strict, retain_failures) |
| `captioning` | VLM and/or LLM sub-sections — captioner type is **inferred from which sub-sections are present** |
| `augmentation` | Model name (enum), executor type, generation parameters, control modalities |
| `evaluators` | Ordered list of hallucination_check, attribute_verification, vlm_verification |

Cross-section validation in `config.py` ensures endpoint requirements match the configured model and captioning strategy. Local executors don't require remote endpoints for cosmos models.

### Factory Dispatch Pattern

Both `captioning/factory.py` and `generation/factory.py` use factory functions that accept the full config and dispatch based on field presence:

- **Captioning** (`create_captioner`): Inferred from which fields are set under `captioning.llm` and `captioning.vlm`. `llm.text` -> TextCaptioner, `llm.file_path` -> FileCaptioner, both vlm+llm -> VLMLLMCaptioner, vlm only -> VLMCaptioner, llm only -> LLMCaptioner. Combining `vlm` with `llm.text`/`llm.file_path` is an error.
- **Generation** (`create_generator`): Dispatches on `augmentation.model.name` enum: `cosmos-transfer2.5`, `cosmos-predict`, `image-edit`.

### Supported Models

| Model | Executor Types | Config Example |
|-------|---------------|----------------|
| `cosmos-transfer2.5` | local, gradio, passthrough | `config_carla_vlm_llm.yaml` |
| `cosmos-predict` | local, gradio | `config_cosmos_predict.yaml` |
| `image-edit` | gradio | `config_image_edit_verification.yaml` |

### Config Validation Flow

`modules/aug_utils/common.py:validate_config_structure()` wraps Pydantic validation. Returns the validated `PipelineConfig` model or `None` on failure. Downstream code still uses raw dicts in some places (accessed via `config_dict`).

### Evaluation Pipeline

Evaluators run in order after generation, with retry on failure (seed incremented each attempt):
1. **Hallucination check** — optical-flow-based motion artifact detection between input/output
2. **Attribute verification** — LLM generates MCQ questions from variables, VLM answers them against the output
3. **VLM verification** — standalone VLM-based quality check

### Module Import Structure

`pyproject.toml` sets `pythonpath = [".", "modules"]`. Modules import each other relative to `modules/` (e.g., `from captioning.factory import create_captioner`, not `from modules.captioning...`).

### Storage Abstraction

All file I/O uses `multistorageclient` (aliased as `msc`) which transparently handles local paths, S3, GCS, Azure, and HTTP URLs. Use `msc.open()` and `msc.is_file()` instead of built-in `open()` for data paths.

### Environment Variables

Endpoints and API keys can be overridden via env vars (e.g., `VLM_ENDPOINT_URL`, `LLM_API_KEY`). Env vars take precedence over config values. `VLM_API_KEY` and `LLM_API_KEY` are the standardized key names.
