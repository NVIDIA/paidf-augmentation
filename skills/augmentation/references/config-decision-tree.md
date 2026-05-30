# Config Decision Tree

## Step 1: Determine the Pipeline from Input Type

**Ask the user what their input is.** The input type determines which model to use:

```
What is your input?
│
├── Video file (.mp4, .avi, etc.)
│   └── Do you want to change scene attributes (weather, lighting, colors, clothing)?
│       ├── Yes → Cosmos Transfer 2.5 (video → augmented video)
│       └── No, I want to predict/continue/extend the video
│           └── Cosmos Predict 2.5 (video + text → new video)
│
├── Image file (.png, .jpg, etc.)
│   └── Do you want to edit the image (change attributes)?
│       ├── Yes → image edit (image → edited image)
│       └── No, I want to generate video from this image
│           └── Cosmos Predict 2.5 (image + text → video)
│
├── Text only (no media input)
│   └── Cosmos Predict 2.5 (text → video)
│
└── Not sure / multiple inputs
    └── Ask the user to clarify their input type and goal
```

## Step 2: Choose a Config

### Cosmos Transfer 2.5 (video → video)

```
├── First-run smoke test        → config_starter.yaml
├── No evaluation needed        → config_carla_vlm_llm.yaml
├── Multi-modal controls        → config_real_outdoor.yaml
├── Hallucination + Attribute   → config_isaacsim.yaml
├── Attribute verification only → config_verification.yaml
```

### Cosmos Predict 2.5 (text/image/video → video)

No dedicated config yet — create one based on any existing config and set `augmentation.model.name: cosmos-predict`.
Supports 3 input modes: text only, image + text, or video + text.

### image edit (image → image)

```
├── With verification                   → config_image_edit_verification.yaml
└── AOI (align output back to input)    → config_image_edit_aoi.yaml
```

### Batch Config Generation

```
└── Generate N configs          → workflow_example.yaml
```

### By Captioning Strategy

```
What model are you using?
│
├── Cosmos Transfer / Predict (video models)
│   └── Default: VLM+LLM (VLM describes scene → LLM generates prompt with variables)
│       Config: config_carla_vlm_llm.yaml, config_isaacsim.yaml
│
├── image edit (image editing)
│   └── Default: LLM-only (LLM generates editing instruction from variables)
│       Config: config_image_edit_verification.yaml
│
└── Fallbacks (any model):
    ├── Fixed text with variable substitution (no endpoints needed)
    │   └── Use captioning.llm.text: "change {color} to {value}"
    ├── Sample from a prompt file (no endpoints needed)
    │   └── Use captioning.llm.file_path: "/path/to/prompts.txt"
    └── VLM-only description (no LLM needed)
        └── Set captioning.vlm only (no captioning.llm)
```

## Endpoint Requirements by Configuration

| Model | Executor | Required Endpoints |
|-------|----------|-------------------|
| cosmos-transfer2.5 | local | None (runs via torchrun subprocess) |
| cosmos-transfer2.5 | gradio | `endpoints.cosmos_transfer` |
| cosmos-transfer2.5 | passthrough | `endpoints.cosmos_transfer` |
| cosmos-predict | local | None |
| cosmos-predict | gradio | `endpoints.cosmos_predict` |
| image-edit | gradio | `endpoints.image_edit` |

Additionally, captioning always needs:
- VLM captioning → `endpoints.vlm`
- LLM captioning → `endpoints.llm` (except text/file captioners which need no endpoint)
- Attribute verification → `endpoints.vlm` + `endpoints.llm` (LLM for question gen, VLM for answering)

## Switching Models Without Changing Configs

Use OmegaConf CLI overrides to switch models on the fly:

```bash
# Switch from cosmos-transfer to image-edit
uv run modules/cli.py --config configs/config_carla_vlm_llm.yaml \
  augmentation.model.name=image-edit \
  augmentation.model.executor_type=gradio \
  endpoints.image_edit.url=http://localhost:8002/v1 \
  endpoints.image_edit.model=Qwen/Qwen-Image-Edit-2511

# Switch executor type (local → gradio)
uv run modules/cli.py --config configs/config_isaacsim.yaml \
  augmentation.model.executor_type=gradio \
  endpoints.cosmos_transfer.url=http://remote-server:30002/ \
  endpoints.cosmos_transfer.model=nvidia/Cosmos-Transfer2.5-7B
```

## Disabling Evaluators Inline

Instead of switching to a config without evaluators, override to disable:

```bash
# Disable hallucination check but keep attribute verification
uv run modules/cli.py --config configs/config_isaacsim.yaml \
  evaluators.0.hallucination_check.enabled=false

# Disable all evaluators (useful for quick generation test)
uv run modules/cli.py --config configs/config_isaacsim.yaml \
  evaluators=null
```

## Multi-Sample Batch Processing

### Inline (up to ~5 samples)

```bash
uv run modules/cli.py --config configs/config_carla_vlm_llm.yaml \
  data.0.inputs.rgb=/data/video1.mp4 \
  data.0.output.video=/output/video1.mp4 \
  data.0.output.caption=/output/video1.txt \
  data.0.output.metadata=/output/video1.json \
  data.1.inputs.rgb=/data/video2.mp4 \
  data.1.output.video=/output/video2.mp4 \
  data.1.output.caption=/output/video2.txt \
  data.1.output.metadata=/output/video2.json
```

### Batch Config Generation (10+ samples)

Use the workflow generator to create per-sample configs from a distribution:

```bash
uv run modules/config_distribution_generation/generate_augmentation_configs.py \
  --workflow configs/workflow_example.yaml
```

The workflow config (`workflow_example.yaml`) specifies:
- `data_dir` — directory of input media files
- `n_augmentations` — number of augmented configs to generate per input
- `config_output` — where to write generated configs
- `example_augmentation_config` — base config template
- `variables` — independent probability distributions for sampled attributes
- `conditional_variables` — dependent attribute distributions keyed by a parent variable value

Use `conditional_variables` when attributes are not independent, for example:
- `road_condition` depends on `weather`
- `shoe_color` depends on `shoe_type`

## Control Modality Configuration (Cosmos Transfer Only)

Control modalities define structural guidance from input videos. Higher weights = more structural preservation from the control input; a weight of 0 (or omitting the key) means the modality is not used.

| Modality | Purpose / recommended use | Weight guidance |
|----------|---------------------------|-----------------|
| `edge` | Edge detection map — preserves structural outlines; the base structure signal in multi-control setups | 0.4–1.0 (use as the foundation) |
| `depth` | Depth estimation map — preserves spatial layout | 0.3–0.4 (moderate) |
| `seg` | Segmentation map — preserves object/semantic boundaries; pair with masks and `seg_control_prompt` | 0.2–0.4 (moderate); never use alone |
| `vis` | Visualization/appearance — preserves visual style, supplements edge/seg | 0.05–0.6 (low); avoid very high values |

### Weight Normalization Behavior

Control weights are **not** individually capped. Instead, normalization depends on the **sum** of all active weights:

- **Sum ≤ 1.0**: Weights are applied **as-is** with no normalization. E.g., `{seg: 0.2, edge: 0.2}` stays unchanged.
- **Sum > 1.0**: Weights are **normalized proportionally** so the total equals 1.0. E.g., `{seg: 4.0, edge: 1.0}` (sum 5.0) becomes `{seg: 0.8, edge: 0.2}`.

**Best practices** (see [Cosmos Cookbook — Control Modalities](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/control_modalities/overview.html)):
- Use multi-control combinations (e.g., edge + seg) for best results
- Start with lower `vis` weights and increase as needed
- Do not use `seg` alone — always pair with another modality
- If the output doesn't incorporate prompt-described changes, increase `augmentation.parameters.guidance` (start at 3, boost to 5+ if needed)

### Control Inputs in Data Section

Control input files must match modalities configured in `augmentation.modalities`:

```yaml
data:
  - inputs:
      rgb: "/path/to/source.mp4"
      controls:
        edge: "/path/to/edge_map.mp4"     # Only needed if modalities.edge is set
        depth: "/path/to/depth_map.mp4"
    output:
      video: "/path/to/output.mp4"
      caption: "/path/to/prompt.txt"
      metadata: "/path/to/metadata.json"

augmentation:
  modalities:
    edge: 0.8
    depth: 0.4
    positive_prompt: "cinematic, photorealistic..."
    negative_prompt: "cartoon, low quality..."
```

If `controls` are set to `null` (e.g., `edge: null`), the Cosmos model extracts control signals automatically from the RGB input.

## Post-Processing: Data-Processing Alignment (image edit only)

`data_processing` is a top-level section (peer of `augmentation` and
`evaluators`) for post-processors that mutate the augmentation output in
place. Each sub-key is a separate processor; presence enables it.

`alignment` runs an MI-registration to warp+crop a generated image back into
the reference frame of the input — useful when the model outputs at a
different resolution (e.g. Qwen Image Edit upscales 209×118 → 1376×768). The
aligned image overwrites `data.output.video`; recovered transform parameters
land in `data.output.metadata` under `alignment`.

```yaml
data_processing:
  alignment:
    rot_range_deg:                     # rotation search [start, stop, step] in degrees
      - -1.0
      - 1.0
      - 0.1
    shift_step:    1                   # tx/ty grid step (px)
    bins:          64                  # MI histogram bins
    interp:        bilinear            # nearest | bilinear warp kernel
    no_resize:     true                # skip pre-resize of align→ref
    min_mi:        null                # null = no MI floor; else abort threshold
```

**Auto-derived parameters — do NOT set these in YAML.**

The following are computed at runtime from the actual image dimensions and
should be left unset. They appear in the schema only as escape-hatch
overrides for the rare case where alignment output is visibly wrong; they are
not knobs to tune by default.

| Field | Auto rule |
|-------|-----------|
| `sx_range` | `[lo, hi, 0.1]` with `lo = ratio*0.9` and `hi = ratio*1.1`, where `ratio = gen_w / ref_w` |
| `sy_range` | `[lo, hi, 0.1]` with `lo = ratio*0.9` and `hi = ratio*1.1`, where `ratio = gen_h / ref_h` |
| `shift_range` | `max(ref_h, ref_w) // 4` |
| `pyr_levels` | `min(3, max_usable)` keeping smallest level ≥ 32 px |

**Agent guidance:** when authoring or editing an alignment config, leave the
four fields above unset. The math rules are well-defined and any explicit
value is more likely to be wrong than the auto-derived one. The only time to
override is when alignment is *visibly failing* on a real run — typical
symptoms: `mi_after` close to `mi_before`, scale recovery saturating at the
range bound, or the aligned image looks shifted/scaled vs the input. In those
cases, copy the auto-logged values from the run and adjust the bounds.

**Requirements:** runs on GPU only (cupy + CUDA).
