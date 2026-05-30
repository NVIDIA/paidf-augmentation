# Captioning Strategy Guide

## Overview

The captioning module produces the text prompt that drives generation. The captioner type is **inferred from field presence** — there is no explicit `type` field.

When evaluator-driven retries are enabled, `pipeline.regenerate_caption_on_retry: true` causes the pipeline to rerun captioning before each retry attempt after the seed is incremented. The prompt file at `output.caption` is overwritten with the most recent prompt.

Dispatch logic in `modules/captioning/factory.py`:

```
1. llm.text is set?          → TextCaptioner     (static template, no AI)
2. llm.file_path is set?     → FileCaptioner      (random selection, no AI)
3. Both vlm + llm present?   → VLMLLMCaptioner    (recommended for video models)
4. Only vlm present?         → VLMCaptioner
5. Only llm present?         → LLMCaptioner        (recommended for image editing)
6. Neither?                  → Error
```

**Recommended defaults by model:**
- **Cosmos Transfer / Predict** (video) → Strategy 1: VLM+LLM
- **image edit** (image) → Strategy 2: LLM-only

**Invalid combinations** (raise `ValueError`):
- `vlm` + `llm.text` — text captioner is standalone, VLM output would be discarded
- `vlm` + `llm.file_path` — same reason
- `llm.text` + `llm.file_path` — mutually exclusive

## Strategy 1: VLM+LLM (Recommended for Most Use Cases)

The VLM describes the input media, then the LLM uses that description plus target variables to generate an editing prompt.

**Flow**: Input media → VLM caption → LLM + variables → Final prompt

**When to use**: When the prompt should reference actual scene content (weather, objects, layout) while changing specific attributes.

**Parser modes**: The VLM supports two parser modes:
- `"instruct"` — uses only `user_prompt` in the conversation (no system prompt). This is the default.
- `"reasoning"` — uses both `system_prompt` and `user_prompt`, and expects the response wrapped in `<answer>` tags.

```yaml
captioning:
  vlm:
    parser: "instruct"              # "instruct" (uses user_prompt only) or "reasoning" (uses both)
    user_prompt: |
      Describe this traffic scene. Focus on weather, lighting, road conditions,
      vehicles, and pedestrians. Do not suggest edits.
    parameters:
      retry: 1                      # Retry on VLM failure
      temperature: 0.3              # Lower = more deterministic
      top_p: 0.95
      frequency_penalty: 1.05       # Reduce repetition
      max_tokens: 4096
      stream: false
      fps: 4.0                      # Frames per second sampled from video
      max_pixels: 307200            # Max pixel count per frame

  llm:
    system_prompt: |
      You are an expert at writing concise prompts for a video generation model.
      You are given:
      1. A caption describing the source scene.
      2. Attribute-value pairs describing the desired changes.
      Generate a single natural-language prompt that changes the scene.
      Output only a JSON object with a single key "prompt".
    example_prompt: |
      Change the traffic scene to a rainy night setting with wet roads while
      preserving the camera viewpoint and object motion.
    parameters:
      temperature: 0.3
      top_p: 0.95
      max_tokens: 512
      frequency_penalty: 1.05
      presence_penalty: 0
      stream: true
    variables:
      weather_condition: ["raining"]
      lighting_condition: ["night"]
      road_condition: ["wet"]
```

**System prompt file**: Instead of inline `system_prompt`, use a file reference:
```yaml
  llm:
    system_prompt_file: "configs/prompts/my_system_prompt.txt"
```

## Strategy 2: LLM-Only (Default for Image Editing)

The LLM generates a natural-language prompt from variable attributes without seeing the input media, but given an example prompt as a starting point.

**Flow**: Variables → LLM → Final prompt

**When to use**: Default for **image edit** (image editing). The user provides target attributes (e.g., "blue tshirt", "white shorts"), and the LLM generates a natural editing instruction. The edit model already sees the image, so VLM description is unnecessary. Also useful for video models when prompt generation doesn't need scene description.

```yaml
captioning:
  llm:
    system_prompt: |
      You are an expert at writing concise prompts for an image editing model.
      You are given attribute-value pairs describing the target changes.
      Generate a single natural-language instruction that applies the target attributes
      while preserving the rest of the scene/person's appearance.
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
      top_outer_type: ["shirt"]
      bottom_type: ["jeans"]
      bottom_color: ["black"]
    verification_options:
      top_outer_color: ["beige", "blue", "brown", "green", "grey", "orange", "pink", "red", "white", "yellow"]
      top_outer_type: ["camisole", "coat", "hoodie", "shirt", "sweater", "tshirt", "vest"]
      bottom_type: ["dress", "jeans", "leggings", "shorts", "skirt"]
      bottom_color: ["beige", "black", "blue", "brown", "green", "grey", "orange", "red", "white", "yellow"]
```

**Requires**: `endpoints.llm`

## Strategy 3: Text Captioner (Fixed Prompt with Variable Substitution)

A template string with `{variable_name}` placeholders replaced by the first value from each variable list.

**Flow**: Template + variables → String substitution → Final prompt

**When to use**: When no LLM/VLM endpoints are available, when you need a deterministic, exact prompt with no AI variation, or when the task is particularly hard for the model and a carefully hand-tuned prompt produces the best results. For example, in the license plate editing use case, significant time was spent tuning and translating the prompt (including to Chinese) for optimal results — in such cases, the text captioner ensures that exact tested prompt is used every time without AI variation. This is pure template substitution — no AI is involved. For general use cases, prefer LLM-only or VLM+LLM captioning for better prompt quality.

```yaml
captioning:
  llm:
    text: >
      change the person top outer color to {top_outer_color},
      top outer type to {top_outer_type},
      bottom type to {bottom_type},
      bottom color to {bottom_color},
      shoe type to {shoe_type},
      shoe color to {shoe_color}
    variables:
      top_outer_color: ["blue"]
      top_outer_type: ["shirt"]
      bottom_type: ["jeans"]
      bottom_color: ["black"]
      shoe_type: ["boots"]
      shoe_color: ["brown"]
```

**Produces**: `change the person top outer color to blue, top outer type to shirt, bottom type to jeans, bottom color to black, shoe type to boots, shoe color to brown`

**Validation at init time**:
- Placeholders like `{color}` must have a matching key in `variables`
- Extra variables not referenced in the template trigger a warning
- Missing variables referenced by placeholders raise a `ValueError`

**No endpoints required** — this captioner runs entirely locally.

## Strategy 4: File Captioner (Sample from File)

Reads prompts from a text file (one prompt per line) and selects one.

**Flow**: Read file → Select line → Final prompt

**When to use**: When you have a curated set of prompts and want to pick one per run.

```yaml
captioning:
  llm:
    file_path: "/path/to/prompts.txt"
    seed: 42                            # Optional: deterministic selection
```

Example `prompts.txt`:
```
Change the scene to a rainy night with wet roads
Transform the environment to a snowy winter morning
Make the scene a foggy twilight with dim streetlights
```

**No endpoints required** — reads from local or cloud storage (via MSC).

## Strategy 5: VLM-Only

The VLM directly describes the input media. The description becomes the prompt.

**Flow**: Input media → VLM → Final prompt (the description itself)

**When to use**: When you want to use the VLM description directly as the generation prompt, without LLM post-processing.

```yaml
captioning:
  vlm:
    parser: "instruct"              # instruct mode uses only user_prompt
    user_prompt: |
      Generate a detailed visual description of this footage for a video
      generation model.
    parameters:
      temperature: 0.3
      max_tokens: 4096
```

**Requires**: `endpoints.vlm`

## Variables vs Verification Values vs Verification Options

Three related but distinct fields in `captioning.llm`:

| Field | Purpose | Used By |
|-------|---------|---------|
| `variables` | Target attributes for prompt generation | LLMCaptioner, TextCaptioner, metadata |
| `verification_values` | Values to verify against (may differ from generation values) | Attribute verification MCQ |
| `verification_options` | All possible MCQ answer choices per attribute | Attribute verification MCQ options |

**When `verification_values` is not set**, the pipeline falls back to `variables` for verification.

**When `verification_options` is not set**, the pipeline uses `variables` as the option pool.

Example with all three:
```yaml
captioning:
  llm:
    variables:
      top_color: ["blue"]               # Used for prompt generation
    verification_options:
      top_color: ["red", "blue", "green", "black"]  # MCQ answer choices
```

If you need verification to test different values than what you generated with:
```yaml
    variables:
      top_color: ["blue"]               # Generation target
    verification_values:
      top_color: ["blue"]               # What to verify (usually same)
    verification_options:
      top_color: ["red", "blue", "green", "black"]  # MCQ choices
```

## Endpoint Resolution Order

For each captioner type, endpoints are resolved in this order:

**VLM endpoint**:
1. `VLM_ENDPOINT_URL` / `VLM_ENDPOINT_MODEL` env vars
2. `endpoints.vlm.url` / `endpoints.vlm.model` config values

**LLM endpoint (for captioning)**:
1. `LLM_CAPTION_ENDPOINT_URL` / `LLM_CAPTION_ENDPOINT_MODEL` env vars
2. `LLM_ENDPOINT_URL` / `LLM_ENDPOINT_MODEL` env vars
3. `endpoints.llm.url` / `endpoints.llm.model` config values

**API keys**:
- VLM: `VLM_API_KEY` env var
- LLM: `LLM_API_KEY` env var

## Common Patterns

### Traffic Scene Augmentation (Cosmos Transfer)
VLM+LLM captioning with weather/lighting/road variables:
→ `config_carla_vlm_llm.yaml` or `config_isaacsim.yaml`

### Person Clothing Editing (image edit)
LLM-only captioning with clothing attribute variables + attribute verification:
→ `config_image_edit_verification.yaml`

### Warehouse Scene Augmentation (Cosmos Transfer)
VLM+LLM with warehouse-specific variables + all evaluators:
→ `config_isaacsim.yaml`

### Multi-Modal Outdoor (Cosmos Transfer)
VLM+LLM with edge + depth + seg + vis control modalities:
→ `config_real_outdoor.yaml`
