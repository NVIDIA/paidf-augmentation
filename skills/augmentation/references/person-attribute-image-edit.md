# Person Attribute Image Edit

Use this reference for PAS person-attribute image editing: clothing/accessory edits, Qwen image edit, PAS distribution configs, attribute verification, single-ID grids, attribute-value tables, and augmented PAS dataset creation.

This is an image-to-image workflow. Use the `image-edit` model path only; do not switch to Cosmos Predict or Cosmos Transfer to solve PAS image-edit requests.

## Core Rules

- Do not modify repo code unless the user explicitly asks. Runtime config copies, logs, and outputs can live under `/tmp` or the requested PAS dataset/output directory.
- Never write model keys to config files, logs, or scripts. Pass user-provided keys through environment variables such as `VLM_API_KEY`, `LLM_API_KEY`, `IMAGE_EDIT_API_KEY`, `BUILD_NVIDIA_API_KEY`, `NVIDIA_API_KEY`, and `NVAPIKEY`.
- **Safe Docker defaults.** Run the container as your own user (`--user "$(id -u):$(id -g)"`) and on the default bridge network so outputs are not world-writable and host ports are not exposed. Add `--network host` only when model endpoints are on `localhost`. Never `chmod 777` (or `o+rwX`) host directories on shared or multi-user machines.
- Keep checked-in PAS distribution configs as templates. Make runtime copies with concrete `data_dir`, `config_output`, `output_root`, and template paths.
- Validate generated configs and inspect expected `output.jpg` and `output_metadata.json` files. The CLI can exit successfully even when a specific sample did not produce an image.
- When `modules/cli.py` fails on `cosmos_oss` before dispatch, treat it as an import-time side effect. Use Docker or a process-local no-op `cosmos_oss.init.init_environment` stub; do not change the generation model.
- For remote OpenAI-compatible model endpoints, the augmentation container does not need GPU flags. NVIDIA GPU support is only required when model endpoints are hosted locally on the same machine.

## Deterministic Docker CLI Workflow

Build the augmentation image from the repo root when the user asks for a fresh local image:

```bash
docker build -t paidf-augmentation:latest -f docker/Dockerfile .
```

Launch the container from the repo root with the dataset and output mounted to stable `/app/data` paths:

```bash
HOST_DATA="/absolute/path/to/dataset_root"
HOST_OUT="/absolute/path/to/augmented_out"
mkdir -p "$HOST_OUT"

docker run -it --rm \
  --user "$(id -u):$(id -g)" \
  --network host \
  -e VLM_API_KEY \
  -e IMAGE_EDIT_API_KEY \
  -e LLM_API_KEY \
  -v "$(pwd)/modules:/app/modules:ro" \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$HOST_DATA:/app/data/in:ro" \
  -v "$HOST_OUT:/app/data/out" \
  -w /app \
  --entrypoint /bin/bash \
  paidf-augmentation:latest
```

> **Security & fallbacks.** Running as your own UID (`--user`) lets the container write `$HOST_OUT` without world-writable permissions, and the `:ro` mounts give read-only access to `modules/`/`configs/`, so no `chmod` is needed for them. The example commands below call endpoints on the host's `localhost`, so the recipe uses `--network host` to reach them on Linux; on macOS/Windows use `--add-host host.docker.internal:host-gateway` and address the endpoints as `http://host.docker.internal:<port>` instead. (Port mapping with `-p` only publishes container ports to the host and does **not** grant access to host services.) If all your endpoints are remote URLs, drop `--network host` and use the default bridge network. If a non-default UID cannot use the image's prebuilt environment, drop `--user` and instead make `$HOST_OUT` writable by the container's `nvidia` uid 1000 (e.g. `chown`), reserving `chmod 777` as a last resort on isolated single-user machines only. Always pass API keys via environment variables — never bake them into the config, image, or logs.

Inside the container, all deterministic commands below assume `/app` as the working directory.

## Input Preparation

Preprocess PAS image folders into multi-panel images with:

```bash
uv run python modules/data_processing/combine_panes.py \
  /app/data/in \
  /app/data/out/panes
```

Use `combine_panes.py` because it accepts one or more images per ID and writes both `{person_key}.jpg` and `{person_key}.json` metadata needed for later splitting. The expected input layout is one subdirectory per person ID:

```text
/app/data/in/
├── person_0001/
│   ├── view_a.jpg
│   ├── view_b.jpg
│   └── view_c.jpg
├── person_0002/
│   ├── view_a.jpg
│   └── view_b.jpg
└── ...
```

The pane metadata sidecar records `image_order`, per-view `widths`, `original_resolutions`, `total_width`, and `height`; keep it beside the pane image so post-processing can split generated multi-pane outputs back to per-view crops.

When the user says to use only one image per ID, select a deterministic source image, usually the first sorted image in that ID folder, and document the chosen input path in the final response.

## PAS Distribution Workflow

1. Make a runtime copy of the requested distribution YAML, commonly `configs/pas_distribution_1000_v1.yaml`.
2. Set concrete runtime paths in the copy:

```yaml
data_dir: /path/to/PAS_baseline/combined_imgs_<run_id>
config_output: /path/to/PAS_baseline/generated_configs_<run_id>
output_root: /path/to/PAS_baseline/output_<run_id>
example_augmentation_config: /path/to/PAS image-edit verification template.yaml
```

3. Generate configs:

```bash
.venv/bin/python modules/config_distribution_generation/generate_augmentation_configs.py \
  --workflow /tmp/pas_distribution_<run_id>.yaml
```

If the template named in the distribution config is missing, use the closest existing PAS image-edit verification config or an existing generated PAS image-edit config as the template source, then apply the compatibility fixes below.

## Generated-Config Compatibility

Apply these fixes before running generated configs against the current CLI:

- Set `data[*].output.video` to the same path as `data[*].output.media`.
- If a legacy top-level `attribute_verification` block exists, add `evaluators: [{attribute_verification: ...}]`.
- In the evaluator copy, remove schema-forbidden legacy keys such as `retries` and `accept_if_mvc_passed`.
- Copy `captioning.verification_options` into `captioning.llm.verification_options` so verification questions get the full answer set.
- Set `pipeline.retry: 0` unless the user explicitly wants regeneration retries.
- For `openai/openai/gpt-5.5` VLM verification, use `temperature: 1.0`; that endpoint rejects `temperature: 0.0`.

Validate configs with `PipelineConfig` before launching a batch when the change touches schema-sensitive fields.

## Running Image Edit

Run image-edit augmentation inside `paidf-augmentation:latest` or another environment with repo dependencies. A process-local wrapper is acceptable when the only blocker is the unused `cosmos_oss` import:

```python
import runpy
import sys
import types

pkg = types.ModuleType("cosmos_oss")
init = types.ModuleType("cosmos_oss.init")
init.init_environment = lambda: None
sys.modules["cosmos_oss"] = pkg
sys.modules["cosmos_oss.init"] = init

config_path = sys.argv[1]
sys.argv = ["modules/cli.py", "--config", config_path]
runpy.run_path("/app/modules/cli.py", run_name="__main__")
```

Run each generated config with `PYTHONPATH=/app/modules` and log stdout/stderr per config. Do not pass API keys on the command line; export them and pass with Docker `-e VAR_NAME`.

Docker and UV notes:

- The Docker image may run as user `nvidia` with a UID that cannot write to the host-owned repo mount. If `uv` reports permission errors under `/app/.venv` or `/app/uv.lock`, run Docker with the host UID/GID, set `HOME=/tmp`, set `UV_PROJECT_ENVIRONMENT=/tmp/augmentation-venv`, and use `uv run --frozen`.
- For NVIDIA-hosted OpenAI-compatible VLM endpoints, configure the OpenAI client base URL, for example `https://inference-api.nvidia.com/v1`, not the full `/chat/completions` URL. The OpenAI client appends `/chat/completions` internally.
- The current CLI can exit with status 0 for a config even when generation fails for that sample. Inspect the expected output files.

For a single smoke-test pane, use the verification config directly:

```bash
mkdir -p /app/data/out/augmented_outputs/person_0001/aug_0

uv run modules/cli.py --config configs/config_image_edit_verification.yaml \
  data.0.inputs.rgb=/app/data/out/panes/person_0001.jpg \
  data.0.output.video=/app/data/out/augmented_outputs/person_0001/aug_0/output.jpg \
  data.0.output.caption=/app/data/out/augmented_outputs/person_0001/aug_0/output.txt \
  data.0.output.metadata=/app/data/out/augmented_outputs/person_0001/aug_0/output_metadata.json \
  endpoints.vlm.url=http://localhost:8000/v1 \
  endpoints.vlm.model=Qwen/Qwen3-VL-30B-A3B-Instruct \
  endpoints.llm.url=http://localhost:8001/v1 \
  endpoints.llm.model=Qwen/Qwen2.5-14B-Instruct \
  endpoints.image_edit.url=http://localhost:8002/v1 \
  endpoints.image_edit.model=Qwen/Qwen-Image-Edit-2511
```

For one augmentation per pane, preserve the standard output layout:

```bash
mkdir -p /app/data/out/augmented_outputs

for pane in /app/data/out/panes/*.jpg; do
  id="$(basename "${pane%.jpg}")"
  mkdir -p "/app/data/out/augmented_outputs/${id}/aug_0"

  uv run modules/cli.py --config configs/config_image_edit_verification.yaml \
    data.0.inputs.rgb="$pane" \
    data.0.output.video="/app/data/out/augmented_outputs/${id}/aug_0/output.jpg" \
    data.0.output.caption="/app/data/out/augmented_outputs/${id}/aug_0/output.txt" \
    data.0.output.metadata="/app/data/out/augmented_outputs/${id}/aug_0/output_metadata.json" \
    endpoints.vlm.url=http://localhost:8000/v1 \
    endpoints.vlm.model=Qwen/Qwen3-VL-30B-A3B-Instruct \
    endpoints.llm.url=http://localhost:8001/v1 \
    endpoints.llm.model=Qwen/Qwen2.5-14B-Instruct \
    endpoints.image_edit.url=http://localhost:8002/v1 \
    endpoints.image_edit.model=Qwen/Qwen-Image-Edit-2511
done
```

To sweep a specific wardrobe value, add OmegaConf list overrides to a single call. Each call samples one combination from the lists:

```bash
mkdir -p /app/data/out/augmented_outputs/person_0001/aug_1

uv run modules/cli.py --config configs/config_image_edit_verification.yaml \
  data.0.inputs.rgb=/app/data/out/panes/person_0001.jpg \
  data.0.output.video=/app/data/out/augmented_outputs/person_0001/aug_1/output.jpg \
  data.0.output.caption=/app/data/out/augmented_outputs/person_0001/aug_1/output.txt \
  data.0.output.metadata=/app/data/out/augmented_outputs/person_0001/aug_1/output_metadata.json \
  'captioning.llm.variables.top_outer_color=[red]' \
  'captioning.llm.variables.top_outer_type=[cropped jacket]' \
  'captioning.llm.variables.bottom_color=[blue]' \
  'captioning.llm.variables.bottom_type=[jeans]' \
  'captioning.llm.variables.shoe_color=[white]' \
  'captioning.llm.variables.shoe_type=[sneakers]'
```

`shoe_type` and `shoe_color` are inserted into the edit prompt but excluded from the default MCQ verification gate by `evaluators[0].attribute_verification.exclude_variables`.

## Output Inspection

Expected files usually have this shape:

```text
input:    <combined_dir>/<person_key>.jpg
output:   <output_root>/<person_key>/aug_<n>/output.jpg
prompt:   <output_root>/<person_key>/aug_<n>/output.txt
metadata: <output_root>/<person_key>/aug_<n>/output_metadata.json
```

Metadata contains `selections`, `attribute_verification.passed`, per-question results, and optional natural captions. A generated image can exist even when verification fails.

Use generated configs as the status source of truth:

```bash
.venv/bin/python -c '
from pathlib import Path
import yaml

config_dir = Path("/path/to/PAS_baseline/generated_configs_<run_id>")
total = done = with_image = with_metadata = metadata_without_image = 0

for cfg in sorted(config_dir.glob("*.yaml")):
    data = yaml.safe_load(cfg.read_text())
    sample = data["data"][0]
    output = Path(sample["output"]["video"])
    metadata = Path(sample["output"]["metadata"])
    image_exists = output.exists()
    metadata_exists = metadata.exists()

    total += 1
    with_image += int(image_exists)
    with_metadata += int(metadata_exists)
    metadata_without_image += int(metadata_exists and not image_exists)
    done += int(image_exists and metadata_exists)

pending = total - done
print(f"total={total} done={done} pending={pending} with_image={with_image} with_metadata={with_metadata} metadata_without_image={metadata_without_image}")
'
```

When the user asks for examples or current status, report concrete paths:

```text
input:    /path/to/PAS_baseline/combined_imgs_<run_id>/00000_RSTP.jpg
output:   /path/to/PAS_baseline/output_<run_id>/00000_RSTP/aug_0/output.jpg
metadata: /path/to/PAS_baseline/output_<run_id>/00000_RSTP/aug_0/output_metadata.json
```

## Augmented PAS Dataset

After image-edit outputs are complete, split outputs back into an augmented PAS dataset with:

```bash
uv run --no-sync python modules/data_processing/create_PAS_augmented_dataset.py \
  --base-dir /app/data/out/panes \
  --augmented-folders /app/data/out/augmented_outputs \
  --output-dir /app/data/out/augmented_dataset \
  --output-json augmented_data.json
```

The post-processing script does not require an original dataset JSON. It uses the pane metadata from preprocessing to split generated multi-pane images back into per-view crops, writes split images under `/app/data/out/augmented_dataset/augmented_imgs/<person_key>_aug<n>/`, and writes the dataset JSON to `/app/data/out/augmented_dataset/augmented_data.json`. The JSON includes selected attributes, generated queries, image paths, and attribute verification metadata.

## Exploratory Grids And Tables

For user-facing examples, start with one ID and one source image unless the user asks for multi-view panels. Generate one attribute modification per call, reuse existing completed cells, and compose grids from the produced `output.jpg` files.

Grid and slide artifact practices:

- Keep one axis or block per attribute and label values directly above or beside the edited image.
- Group attributes visually when the user asks for a wide/horizontal slide artifact.
- Avoid an original column unless the user asks for it.
- Keep tiles close together and remove excess whitespace for slide-ready outputs.
- Save both PNG and JPG when useful for slides.
- For tables of attribute-value pairs, read the relevant PAS distribution/config YAML and output a Markdown or image table without running generation.
- For color vocabulary examples, prefer swatch/table artifacts when no generation is needed; use generated image grids only when the user asks to see edits on the person.
