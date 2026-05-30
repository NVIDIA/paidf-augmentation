---
name: augmentation
description: >-
  Run the NVIDIA Physical AI Data Factory (PAIDF) augmentation pipeline:
  author and validate YAML configs and run inference with Cosmos Transfer 2.5
  (video style transfer), Cosmos Predict 2.5 (text/image/video-to-video), and
  image-edit models — covering captioning, control modalities, evaluators
  (hallucination, attribute, VLM verification), data_processing alignment, and
  PAS person-attribute edits via Docker GPU inference. Trigger keywords:
  augmentation, cosmos transfer, cosmos predict, image edit, PAS person
  attributes, AOI/PCB alignment, captioning, attribute verification, config
  validation. Do NOT use for model training, cluster/NIM deployment, or
  unrelated app/database work.
license: Apache-2.0
metadata:
  owner: NVIDIA
  service: augmentation
  version: 1.0.0
  reviewed: '2026-05-30'
  author: NVIDIA
  tags:
    - physical-ai
    - augmentation
    - cosmos
    - image-edit
---

# PAIDF Augmentation Pipeline Skill

Unified pipeline for augmenting camera data through NVIDIA generative AI models (Cosmos Transfer 2.5, Cosmos Predict 2.5, image edit) with automated captioning, generation, and quality evaluation.

## Purpose

Use this skill to drive the PAIDF augmentation pipeline end to end:

- **Select the right model** for the input and goal — Cosmos Transfer 2.5 (transform a video), Cosmos Predict 2.5 (generate or extend video), or image-edit (edit an image).
- **Author and validate YAML configs** against the `PipelineConfig` Pydantic schema.
- **Configure captioning** (VLM, LLM, text, or file) and **evaluators** (hallucination check, attribute verification, VLM verification).
- **Launch and run** inference inside the `paidf-augmentation:latest` Docker container.

Do **not** use this skill for training models, deploying clusters or NIM endpoints, or unrelated application/database development.

## Instructions

Use this skill when working with the augmentation pipeline: running inference, authoring or editing YAML configs, debugging validation or runtime errors, adding new data samples, configuring captioning strategies, tuning generation parameters, or setting up evaluators. Prefer the documented config patterns and CLI workflows over ad-hoc approaches.

### Use Case References

- **PAS person-attribute image edit**: For PAS folders, person clothing/accessory edits, Qwen image edit, PAS distribution YAMLs, single-ID image grids, attribute-value tables, and augmented PAS dataset creation, read [Person Attribute Image Edit](references/person-attribute-image-edit.md).
- **AOI / PCB image edit with output alignment**: For PCB AOI workflows where the model output is upscaled and must be warped back into the input frame (MI-registration), see the *Post-Processing: Data-Processing Alignment* section of [Config Decision Tree](references/config-decision-tree.md) and example `configs/config_image_edit_aoi.yaml`.
- **General model/config selection**: For non-PAS model routing and config choice, read [Config Decision Tree](references/config-decision-tree.md).

### Model Selection (Ask the User)

When the user wants to run augmentation but hasn't specified a model, determine the right pipeline by asking about their **input type and goal**:

- **Cosmos Transfer 2.5** — video augmentation (style transfer). Takes a video and changes scene attributes (weather, lighting, clothing, colors) while preserving the structure and motion. Think of it as "reskinning" an existing video.
- **Cosmos Predict 2.5** — content generation/extension. Generates new video from text, extends video from a starting frame (image + text), or continues/predicts from existing video (video + text). Think of it as "creating new content" rather than transforming existing content.
- **image edit** — image-to-image editing. Takes a single image and edits specific attributes based on a text instruction.

| Input Type | Pipeline | Model |
|-----------|----------|-------|
| Video — change scene attributes (weather, lighting, style) | Cosmos Transfer 2.5 | `cosmos-transfer2.5` — video-to-video augmentation with control modalities |
| Video + text — extend or predict continuation | Cosmos Predict 2.5 | `cosmos-predict` — video-conditioned prediction/continuation |
| Image — edit specific attributes | image edit | `image-edit` — image-to-image editing |
| Image + text — generate video from a starting frame | Cosmos Predict 2.5 | `cosmos-predict` — image-conditioned video generation |
| Text only — generate video from scratch | Cosmos Predict 2.5 | `cosmos-predict` — text-to-video generation |

**Key rule**: If the input is a video and the user wants to **change scene attributes** (weather, lighting, clothing), use **Cosmos Transfer**. If the user wants to **predict/generate new video** from text, image, or video conditioning, use **Cosmos Predict**.

See [Config Decision Tree](references/config-decision-tree.md) for detailed decision logic and config recommendations.

## Before You Start

### Step 1: Prerequisites

Before launching the container, verify the following with the user:

1. **Docker with NVIDIA runtime** — required for all inference:
   ```bash
   docker --version     # Must have NVIDIA container runtime installed
   ```

2. **HuggingFace token (`HF_TOKEN`)** — required for Cosmos model downloads. The `nvidia/Cosmos-*` repos are gated under the NVIDIA Open Model License; the user must accept the license once on each repo they plan to use (e.g. `nvidia/Cosmos-Transfer2.5-2B`, `nvidia/Cosmos-Predict2.5-2B`) on huggingface.co before their token works. Confirm the user's `HF_TOKEN` and that the licenses are accepted.

3. **VLM and LLM endpoints** — required for captioning and evaluation (unless using text/file captioner). Default models are already configured in the example configs (`Qwen/Qwen3-VL-30B-A3B-Instruct` for VLM, `Qwen/Qwen2.5-14B-Instruct` for LLM). If the user does **not** already have running VLM/LLM containers, ask them to provide endpoint URLs or help them set up the endpoints.

4. **Generation model endpoints** — depends on the model and executor type:
   - **Cosmos Transfer 2.5** — runs **locally** by default (`executor_type: local`), no remote endpoint needed
   - **Cosmos Predict 2.5** — runs **locally** by default (`executor_type: local`), no remote endpoint needed
   - **image edit** — runs via **remote endpoint** (`executor_type: gradio`), requires `endpoints.image_edit` to be configured

### Step 2: Launch the Docker Container (First Step for All Inference)

All inference runs inside the `paidf-augmentation:latest` Docker container. This is always the first step — the host environment does not have the required model dependencies (e.g., `cosmos_transfer2`).

Launch the container interactively from the project root:

```bash
docker run -it --rm \
  --network host \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e HF_TOKEN=<user's HF token> \
  -e HF_HOME=/home/nvidia/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/home/nvidia/.cache/huggingface/hub \
  -v "$HOME/.cache/huggingface:/home/nvidia/.cache/huggingface" \
  -v "$(pwd)/modules:/app/modules" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/data:/app/data" \
  --entrypoint /bin/bash \
  paidf-augmentation:latest
```

**Optional — pre-configured endpoints and cloud storage:** add `--env-file examples/.env` and `-v "$(pwd)/examples/secrets.json:/var/secrets/secrets.json:ro"` to the command above. Both are only needed if you already have endpoint URLs or cloud storage credentials; runs from scratch with local executors and local file paths can omit them.

> **Security:** `--network host` removes network isolation, and `HF_TOKEN`/API keys are passed via environment variables. Review [Security Notes](#security-notes) before running on a shared or production host.

Key volume mounts:
- `modules/` and `configs/` are mounted from the host, so edits take effect without rebuilding
- `data/` is mounted for pipeline output (videos, captions, metadata) — it must be writable by the container user (see [Security Notes](#security-notes))
- the host HuggingFace cache (`$HOME/.cache/huggingface`) is mounted at the container's `HF_HOME` to avoid re-downloading models between runs
- the optional `secrets.json` mount is read-only and supplies cloud storage credentials

### Step 3: Run the Pipeline (Inside the Container)

Once inside the container, run inference with:

```bash
uv run modules/cli.py --config configs/<config_file>.yaml

# With OmegaConf CLI overrides (dot-list syntax)
uv run modules/cli.py --config configs/config_isaacsim.yaml \
  data.0.inputs.rgb=/path/to/video.mp4 \
  augmentation.parameters.seed=42
```

### Environment Variables

Endpoints and API keys can be set via env vars (take precedence over config values):

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | **Required.** HuggingFace token for model downloads |
| `VLM_API_KEY` | API key for VLM endpoints |
| `LLM_API_KEY` | API key for LLM endpoints |
| `VLM_ENDPOINT_URL` | Override VLM endpoint URL |
| `LLM_ENDPOINT_URL` | Override LLM endpoint URL |
| `COSMOS_ENDPOINT_URL` | Override Cosmos Transfer endpoint |
| `COSMOS_PREDICT_ENDPOINT_URL` | Override Cosmos Predict endpoint |
| `IMAGE_EDIT_ENDPOINT_URL` | Override image edit endpoint |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Configuration Schema

All YAML configs are validated against the `PipelineConfig` Pydantic model (`modules/aug_utils/schema/`) and have seven top-level sections: `data` (input/output paths), `endpoints`, `pipeline` (retry/logging/eval settings), `captioning`, `augmentation` (model + parameters + control modalities), `data_processing` (optional post-processors), and `evaluators`. For full per-section YAML examples, read [configuration-schema.md](references/configuration-schema.md).


## Supported Models

| Model | Name Enum | Input | Output | Executor Types | Endpoint Key |
|-------|-----------|-------|--------|----------------|-------------|
| Cosmos Transfer 2.5 | `cosmos-transfer2.5` | Video + control modalities | Video | local, gradio, passthrough | `cosmos_transfer` |
| Cosmos Predict 2.5 | `cosmos-predict` | Text, Image+Text, or Video+Text | Video | local, gradio | `cosmos_predict` |
| image edit | `image-edit` | Image | Image | gradio | `image_edit` |

Cosmos Predict's input modes are text-only, image+text (start frame), or video+text (extend/predict). The **local** executor runs via a `torchrun` subprocess (multi-GPU, no remote endpoint); the **gradio** executor calls a remote Gradio API endpoint. See *Model Selection* above for how to choose a model from the input type and goal.

## Pipeline Flow

```
1. Load & validate config (Pydantic PipelineConfig)
2. Initialize captioner (factory dispatch from config)
3. Initialize evaluators (hallucination, attribute verification)
4. Initialize generator (factory dispatch from augmentation.model.name)
5. For each data sample:
   a. Run captioning -> produce prompt
   b. Run generation (with seed) -> produce output media
   c. Run hallucination check (if configured)
   d. Run attribute verification (if configured)
      - LLM generates MCQ questions from variables
      - VLM answers questions against output media
   e. On failure: retry with incremented seed (up to `pipeline.retry`)
      - If `pipeline.regenerate_caption_on_retry: true` and a captioner is configured, rerun captioning and overwrite the prompt saved to `output.caption` before retrying generation
   f. Write metadata JSON
```

## Examples

### Example configs

| Config | Model | Use Case |
|--------|-------|----------|
| `config_starter.yaml` | cosmos-transfer2.5 | Minimal first-run on bundled `data/sample_input.mp4` (hallucination check) |
| `config_carla_vlm_llm.yaml` | cosmos-transfer2.5 | Traffic scene augmentation (no evaluators) |
| `config_isaacsim.yaml` | cosmos-transfer2.5 | Warehouse scene + hallucination/attribute/VLM verification |
| `config_verification.yaml` | cosmos-transfer2.5 | Attribute + VLM verification only |
| `config_image_edit_verification.yaml` | image-edit | Person clothing editing + verification |
| `config_image_edit_aoi.yaml` | image-edit | AOI PCB inference with post-output MI alignment (`data_processing.alignment`) |
| `config_real_outdoor.yaml` | cosmos-transfer2.5 | Multi-modal outdoor scenes |
| `workflow_example.yaml` | N/A | Batch config generation with dependent attribute sampling |

Captioning and evaluator details for each config are in [config-decision-tree.md](references/config-decision-tree.md).

### Worked example: re-render a video as a rainy night

1. **Model:** input is a video and the goal is a scene-attribute change → Cosmos Transfer 2.5. Start from `config_isaacsim.yaml` (`augmentation.model.name: cosmos-transfer2.5`, default `executor_type: local`).
2. **Inputs/outputs:** point `data.0.inputs.rgb` at the source video and set `data.0.output.{video,caption,metadata}`.
3. **Target attributes:** set `captioning.llm.variables.weather: ["rainy"]` and `time_of_day: ["night"]`.
4. **Run** (inside the container, see Step 2): `uv run modules/cli.py --config configs/config_isaacsim.yaml`.

## Common Tasks

### Add a New Data Sample

Add an entry to the `data` list in any config:

```yaml
data:
  - inputs:
      rgb: "/path/to/new_video.mp4"
    output:
      video: "/path/to/output.mp4"
      caption: "/path/to/prompt.txt"
      metadata: "/path/to/metadata.json"
```

### Change Target Attributes

Edit `captioning.llm.variables` — each key maps to a list of values (first value is used for text captioner, all values available for LLM generation):

```yaml
captioning:
  llm:
    variables:
      weather: ["snowy"]
      time_of_day: ["dusk"]
```

For attribute verification, also set `verification_options` with all possible MCQ answer choices:

```yaml
    verification_options:
      weather: ["clear", "cloudy", "rainy", "snowy"]
```

### Generate Batch Configs With Dependencies

For many configs from a workflow YAML, use `conditional_variables` for attributes that depend on another — e.g. `road_condition` depends on `weather`, so snowy weather never pairs with dry roads. This avoids invalid combinations you'd get from independent distributions. See [Config Decision Tree](references/config-decision-tree.md) for the full `workflow_example.yaml` structure and the generator command.

### Switch Between Models

Change `augmentation.model.name` and add the matching endpoint (e.g. `image-edit` needs `executor_type: gradio` plus `endpoints.image_edit` with a `url` and `model`). You can also override these on the CLI without editing the file; see [Config Decision Tree](references/config-decision-tree.md).

### Tune Generation Quality

Key parameters for Cosmos Transfer:
- `augmentation.parameters.sigma` — noise level (higher = more change, default 90)
- `augmentation.parameters.guidance` — prompt adherence (default 3.0)
- `augmentation.parameters.num_steps` — denoising steps (default 35, more = slower but better)
- `augmentation.modalities.edge` — edge control weight (1.0 = strict structure preservation)

### Run Tests (Inside the Container)

```bash
uv run pytest tests/                          # all tests
uv run pytest tests/schema/test_schema.py -v  # schema tests only
```

Tests and all `uv run` commands must run inside the container (Step 2); the host lacks the required dependencies.

## Troubleshooting

All inference and schema validation must run **inside the Docker container** (Step 2). Import errors such as `ModuleNotFoundError: No module named 'cosmos_transfer2'` or `COSMOS local execution is not supported in this environment` mean you are running on the host. For config-validation errors, runtime errors, and typical per-stage timings, see [troubleshooting.md](references/troubleshooting.md).

## Storage

All file I/O uses `multistorageclient` (aliased as `msc`) which transparently handles local paths, S3 (`s3://`), GCS (`gs://`), Azure (`az://`), and HTTP URLs. Configure cloud storage via `examples/secrets.json` mounted at `/var/secrets/secrets.json`.

## Security Notes

This skill runs GPU inference in Docker with credentials and host networking. Apply these practices, especially on shared or production hosts:

- **Credentials via environment variables.** `HF_TOKEN`, `VLM_API_KEY`, `LLM_API_KEY`, and cloud storage secrets are passed as env vars or env-files. Never hardcode them in config YAML, logs, or scripts, and never commit them. Treat `HF_TOKEN` as sensitive — it grants access to gated model repos. Give any local `.env` file restrictive permissions (`chmod 600 examples/.env`), keep it out of version control, and avoid echoing or logging secret values. For production, prefer a secrets manager or Docker secrets over an on-disk env-file.
- **Data-directory permissions.** `data/` must be writable by the container user (e.g. `nvidia`, uid 1000). Prefer matching the host directory's ownership to that uid (`sudo chown -R 1000:1000 data/`) or running the container as your own user (`--user "$(id -u):$(id -g)"`). If you must loosen permissions, use group write with proper group membership (`chmod 775 data/`). `chmod 777` is a last resort and must never be used on shared or production systems — a world-writable directory lets any local user or process tamper with pipeline outputs.
- **Host networking.** The launch recipe uses `--network host` so the container can reach model endpoints on the host's `localhost` (on macOS/Windows use `--add-host host.docker.internal:host-gateway` and address `http://host.docker.internal:<port>` instead). It removes network isolation, so when your endpoints are remote URLs prefer the default bridge network, and avoid host networking on shared or production hosts. Port mapping (`-p`) only publishes container ports to the host and does not grant access to host services.

## Limitations

- **GPU + Docker required.** All inference, schema validation, and `uv run`/`pytest` commands run inside the `paidf-augmentation:latest` container; the host lacks the model dependencies. The `data_processing.alignment` post-processor additionally requires a CUDA GPU (cupy).
- **Inference only.** This pipeline augments and generates media — it does not train or fine-tune models.
- **Gated models.** Cosmos weights require an `HF_TOKEN` with the NVIDIA Open Model License accepted on each `nvidia/Cosmos-*` repo.
- **image-edit is remote-only.** The `image-edit` model runs via a Gradio endpoint (`endpoints.image_edit`); there is no local executor for it.

## Reference files

See [references/README.md](references/README.md) for the full index of reference docs (configuration schema, config decision tree, captioning strategies, evaluator setup, troubleshooting, and PAS person-attribute image edit).
