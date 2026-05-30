# Configuration Schema

All YAML configs are validated against a Pydantic `PipelineConfig` model (`modules/aug_utils/schema/`). Seven top-level sections:

```yaml
data:             # List of input/output sample paths (required, min 1)
endpoints:        # API endpoint URLs and models
pipeline:         # Retry, logging, evaluation settings
captioning:       # VLM and/or LLM captioning configuration
augmentation:     # Model selection, parameters, modalities
data_processing:  # Optional post-processors (e.g. alignment) — runs after generation
evaluators:       # Ordered list of quality checks
```

## `data` — Input/Output Samples

```yaml
data:
  - inputs:
      rgb: "/path/to/input.mp4"          # Required: source video or image
      controls:                           # Optional: control modality inputs
        edge: "/path/to/edge.mp4"
        depth: "/path/to/depth.mp4"
        seg: "/path/to/seg.mp4"
        vis: "/path/to/vis.mp4"
    output:
      video: "/path/to/output.mp4"        # Required: generated output
      caption: "/path/to/prompt.txt"       # Required: generated prompt text
      metadata: "/path/to/metadata.json"   # Required: run metadata
      evaluation: "/path/to/eval.json"     # Optional: evaluation results
```

## `endpoints` — API Endpoints

```yaml
endpoints:
  vlm:
    url: "http://localhost:8000/v1"
    model: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  llm:
    url: "http://localhost:8001/v1"
    model: "Qwen/Qwen2.5-14B-Instruct"
  cosmos_transfer:
    url: "http://localhost:30002/"
    model: "nvidia/Cosmos-Transfer2.5-7B"
  cosmos_predict:
    url: "http://localhost:30003/"
    model: "nvidia/Cosmos-Predict2.5-7B"
  image_edit:
    url: "http://localhost:8002/v1"
    model: "Qwen/Qwen-Image-Edit-2511"
```

Cross-section validation enforces: captioning with VLM requires `endpoints.vlm`, captioning with LLM requires `endpoints.llm` (unless using text/file captioner), generation model requires its corresponding endpoint (unless `executor_type: local`).

## `pipeline` — Pipeline Settings

```yaml
pipeline:
  retry: 1                           # Max retries on evaluation failure (default: 1, keep low to save time)
  regenerate_caption_on_retry: true  # On evaluator failure, bump seed and rerun captioning before retrying generation
  logging:
    enabled: true
    level: "INFO"                     # DEBUG, INFO, WARNING, ERROR
  evaluation:
    strict: true                      # Fail sample on any evaluator failure
    retain_failures: true             # Keep output files on failure
```

## `captioning` — Captioning Strategies

Captioner type is **inferred from which sub-sections and fields are present** (no explicit `type` field):

| Config Pattern | Captioner | Description |
|---------------|-----------|-------------|
| `llm.text` set | TextCaptioner | Fixed prompt with `{variable}` substitution |
| `llm.file_path` set | FileCaptioner | Sample prompts from a text file |
| Both `vlm` + `llm` | VLMLLMCaptioner | VLM describes media, LLM generates prompt |
| Only `vlm` | VLMCaptioner | VLM-only description |
| Only `llm` (with variables) | LLMCaptioner | LLM generates prompt from variables |

**Invalid combination**: `vlm` + `llm.text` or `vlm` + `llm.file_path` will raise an error.

### VLM+LLM Captioning (most common)

**Important**: The `system_prompt` and `user_prompt` must be tailored to the scenario. When switching between scenarios (e.g., traffic → warehouse → person editing), update these prompts accordingly. Below are examples for each common scenario.

**Traffic scene** (Cosmos Transfer — weather/lighting changes):
```yaml
captioning:
  vlm:
    parser: "instruct"
    system_prompt: |
      You are a helpful assistant that describes traffic-scene video content.
      You MUST ONLY describe the scene itself, never video quality or technical artifacts.
      Respond with plain descriptive text only.
    user_prompt: |
      Describe this traffic intersection surveillance footage.
      Focus on weather, time of day, lighting, road condition, vehicles, pedestrians,
      traffic signals, and the static environment.
      Do not suggest edits. Only describe what is currently visible in the source footage.
    parameters:
      temperature: 0.3
      top_p: 0.95
      frequency_penalty: 1.05
      max_tokens: 4096
      stream: false
      fps: 4.0
      max_pixels: 307200
  llm:
    system_prompt: |
      You are an expert at writing concise prompts for a video generation model.
      You are given:
      1. A caption describing the source traffic scene.
      2. Attribute-value pairs describing the desired target conditions.
      Generate a single natural-language prompt that changes the scene to match the
      target attributes while preserving viewpoint, scene layout, vehicle motion,
      and object consistency.
      Output only a JSON object with a single key "prompt" containing the final sentence.
    example_prompt: |
      Change the traffic scene to a rainy night setting with wet roads while
      preserving the original camera viewpoint, traffic layout, and object motion.
    parameters:
      temperature: 0.3
      top_p: 0.95
      max_tokens: 512
      frequency_penalty: 1.05
      stream: true
    variables:
      weather_condition: ["raining"]
      lighting_condition: ["night"]
      road_condition: ["wet"]
```

**Warehouse scene** (Cosmos Transfer — environment changes):
```yaml
captioning:
  vlm:
    parser: "instruct"
    system_prompt: |
      You are a helpful assistant. Respond with plain descriptive text only.
    user_prompt: |
      Analyze the warehouse CCTV footage and generate a detailed description of the
      visual elements in the scene. Focus on the physical environment (e.g., warehouse
      layout, shelves, floor, equipment, walls, etc.), the objects (such as boxes,
      forklifts, workers), and the events (e.g., boxes falling, workers positions,
      gestures, reactions, forklift moving). Use objective, neutral language — do not
      invent a story or speculate about causes or intentions.
    parameters:
      temperature: 0.3
      top_p: 0.95
      frequency_penalty: 1.05
      max_tokens: 4096
      stream: false
      fps: 4.0
      max_pixels: 307200
  llm:
    system_prompt: |
      You are an expert at writing concise prompts for a video generation model.
      You are given:
      1. A caption describing the source warehouse scene.
      2. Attribute-value pairs describing desired visual changes.
      Generate a single natural-language prompt that applies the target attributes
      while preserving the original scene layout and object positions.
      Output only a JSON object with a single key "prompt" containing the final sentence.
    parameters:
      temperature: 0.3
      top_p: 0.95
      max_tokens: 512
      frequency_penalty: 1.05
      stream: true
    variables:
      shadow_intensity: ["strong"]
```

**Person/clothing editing** (image edit — LLM-only captioning):
```yaml
captioning:
  llm:
    system_prompt: |
      You are an expert at writing concise prompts for an image editing model.
      You are given attribute-value pairs describing the target clothing changes.
      Generate a single natural-language instruction that updates the person's
      clothing to match those target attributes while preserving the rest of the
      person's appearance and identity.
      Output only a JSON object with a single key "prompt" containing the final sentence.
    example_prompt: |
      Change the person's clothing to a blue shirt, with black jeans, and brown boots,
      while preserving the person's identity and keeping the appearance consistent
      across all views.
    parameters:
      temperature: 0.3
      top_p: 0.95
      max_tokens: 512
      frequency_penalty: 1.05
      stream: true
    variables:
      top_outer_color: ["blue"]
      bottom_type: ["jeans"]
      shoe_type: ["boots"]
```

**Traffic/scene image editing** (image edit — LLM-only captioning):
```yaml
captioning:
  llm:
    system_prompt: |
      You are an expert at writing concise prompts for an image editing model.
      You are given attribute-value pairs describing the desired visual changes.
      Generate a single natural-language instruction that applies the target attributes
      while preserving the rest of the scene.
      Output only a JSON object with a single key "prompt" containing the final sentence.
    example_prompt: |
      Replace the red car with a blue truck facing the opposite direction,
      while preserving the rest of the traffic scene and background.
    parameters:
      temperature: 0.3
      top_p: 0.95
      max_tokens: 512
      frequency_penalty: 1.05
      stream: true
    variables:
      vehicle_type: ["truck"]
      vehicle_color: ["blue"]
```

**When switching scenarios**, the key things to update are:

For **VLM+LLM** (video models — Cosmos Transfer/Predict):
1. **VLM `system_prompt`** — what domain the assistant focuses on
2. **VLM `user_prompt`** — what scene elements to describe (traffic signals vs shelves vs clothing)
3. **LLM `system_prompt`** — what kind of model output (video generation prompt vs image editing instruction)
4. **LLM `example_prompt`** — a representative example for the new scenario
5. **LLM `variables`** — the target attributes relevant to the scenario

For **LLM-only** (image editing — image edit):
1. **LLM `system_prompt`** — what kind of editing instruction to generate (clothing changes vs scene edits vs object replacement)
2. **LLM `example_prompt`** — a representative example for the new scenario
3. **LLM `variables`** — the target attributes relevant to the scenario
4. **LLM `verification_options`** — MCQ answer choices for attribute verification (if evaluators enabled)

### Captioning Strategy Selection

When the user specifies target attributes (e.g., "change forklift to blue", "blue tshirt and white shorts"), **prefer AI-generated captioning** over `llm.text`. AI captioners (VLM+LLM or LLM-only) produce more natural, context-aware prompts that lead to better generation quality, whereas `llm.text` is just static template substitution with no AI involved.

**Default by model type:**

| Model | Default Captioning | Why |
|-------|-------------------|-----|
| **Cosmos Transfer 2.5** (video) | VLM+LLM | VLM describes the scene context (layout, objects, weather), LLM incorporates that with target attributes for the best video augmentation prompts |
| **Cosmos Predict 2.5** (video) | VLM+LLM | Same reasoning — scene context improves prediction/generation quality |
| **image edit** (image) | LLM-only | User provides specific target attributes; LLM generates a natural editing instruction from variables. VLM description is unnecessary since the edit model already sees the image |

**Fallback order (when the default isn't possible):**
1. **VLM+LLM** — Best for video models. Requires both `endpoints.vlm` and `endpoints.llm`.
2. **LLM-only** — Default for image editing. Requires only `endpoints.llm`. LLM generates a prompt from variables without seeing the media.
3. **Text captioner (`llm.text`)** — Use **only** when no LLM/VLM endpoints are available, or when you need a deterministic, exact prompt with no AI variation. This is pure template substitution (`{variable}` → value), not AI-generated.

### Text Captioning (fixed prompt with variable substitution)

**Note:** This is a static template — no AI is involved. Use VLM+LLM or LLM-only captioning for better prompt quality when endpoints are available.

```yaml
captioning:
  llm:
    text: "change the person top outer color to {top_outer_color}, bottom type to {bottom_type}"
    variables:
      top_outer_color: ["blue"]
      bottom_type: ["jeans"]
```

Placeholders are validated at init time — missing or extra variables raise errors immediately.

### File Captioning (sample from file)

```yaml
captioning:
  llm:
    file_path: "/path/to/prompts.txt"   # One prompt per line
    seed: 42                            # Optional: reproducible selection
```

## `augmentation` — Model & Generation

```yaml
augmentation:
  model:
    name: "cosmos-transfer2.5"    # cosmos-transfer2.5 | cosmos-predict | image-edit
    executor_type: "local"        # local | gradio | passthrough
    version: "ct2.5"              # Optional version tag
  parameters:
    sigma: 90                     # Cosmos Transfer noise level
    seed: null                    # null = random per run
    guidance: 3.0
    num_steps: 35
    inference_name: "cosmos_transfer_inference"
  local_parameters:
    num_processes: 4              # Number of GPUs for local execution
    master_port: 12341
  modalities:                     # Control modality weights (cosmos-transfer only)
    edge: 1.0
    depth: 0.4
    seg: 0.3
    vis: 0.05
    seg_control_prompt: "road surface, vehicles, sidewalks"
    positive_prompt: "cinematic, photorealistic, ultra high quality..."
    negative_prompt: "cartoon, pixelated, low quality..."
```

## `data_processing` — Post-Processors

Optional post-processors applied to augmentation outputs after generation, in
place. Each sub-key is a separate processor; presence enables it (no `enabled`
flag). The processor mutates the file at `data.output.video` and writes its
parameters into `data.output.metadata` under the same sub-key.

Currently supported:

| Sub-key | Purpose |
|---------|---------|
| `alignment` | 5-DOF affine MI-registration warping the generated image back into the input's frame; required for AOI where the model upscales (e.g. 209×118 → 1376×768). GPU-only (cupy). |

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

**Do NOT set `sx_range`, `sy_range`, `shift_range`, or `pyr_levels`** when
authoring a config. They are auto-derived in `run_alignment` from the
generated and reference image dimensions and should be left unset. Schema
exposes them only as escape-hatch overrides for *visibly failing* alignment
runs. The derivation rules, override criteria, and more detail on the
alignment post-processor live in the *Post-Processing: Data-Processing
Alignment* section of the config decision tree reference.

## `evaluators` — Quality Checks

Evaluators run in order after generation. On failure, the pipeline retries with an incremented seed according to `pipeline.retry`.

```yaml
evaluators:
  - hallucination_check:
      enabled: true
      threshold: 0.682           # Optical flow similarity threshold
      params:
        grad_thresh: 10.0
        blur_ksize: 7
        morph_k: 3
        dist_tol_px: 7.0
  - attribute_verification:
      enabled: true
      generate_natural_caption_on_pass: true
      extra_questions:            # Additional MCQ checks
        - variable: "multi_view_consistency"
          question: "Is the appearance consistent across all views?"
          options:
            A: "Yes, consistent"
            B: "No, inconsistent"
          correct_answer: "A"
          request_reasoning: true
      question_generation:
        system_prompt: "..."
        parameters:
          temperature: 0.2
          max_tokens: 2048
      vlm_verification:           # VLM prompt + params used to answer
        system_prompt: "..."      # the MCQ questions generated above
        parameters:
          temperature: 0.0
          max_tokens: 10
```
