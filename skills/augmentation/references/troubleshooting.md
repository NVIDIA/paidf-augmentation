# Troubleshooting

All inference and schema validation must run **inside the `paidf-augmentation:latest` Docker container** (see SKILL.md → *Before You Start, Step 2*). If you see import errors, you are almost certainly running on the host, which lacks the model dependencies.

## Config Validation Errors

All configs are validated by the `PipelineConfig` Pydantic model. Common issues:

| Error | Cause | Fix |
|-------|-------|-----|
| `augmentation.model.name='cosmos-transfer2.5' ... requires endpoints.cosmos_transfer` | Non-local executor but no endpoint | Add `endpoints.cosmos_transfer` or set `executor_type: local` |
| `captioning.vlm requires endpoints.vlm` | VLM captioning without a VLM endpoint | Add an `endpoints.vlm` section |
| `'text' and 'file_path' are mutually exclusive` | Both set in `captioning.llm` | Use one or the other |
| `captioning.llm.text / file_path cannot be combined with captioning.vlm` | Invalid captioner combination | Use text/file alone, or VLM+LLM without text/file |
| `Template has placeholders {x} not found in variables` | Text template references an undefined variable | Add the missing variable to `captioning.llm.variables` |

## Runtime Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `COSMOS local execution is not supported in this environment` | Running on host or missing cosmos dependencies | Must run inside the container (Step 2) |
| `ModuleNotFoundError: No module named 'cosmos_transfer2'` | Running on host instead of Docker | Must run inside the container (Step 2) |
| Verification fails repeatedly | Generated output doesn't match target attributes | Increase `pipeline.retry`, adjust `augmentation.parameters.guidance`, or tune `augmentation.parameters.sigma` |
| Hallucination check fails | Excessive motion artifacts in output | Lower `augmentation.parameters.sigma`, or raise `evaluators[].hallucination_check.threshold` |

## Typical Inference Timeline

Approximate per-sample timing from production runs:

| Stage | image edit | Cosmos Transfer (4 GPU) |
|-------|------------|-------------------------|
| Config validation + captioner init | <2s | <2s |
| VLM+LLM captioning | ~3s | ~6s |
| Generation | ~1.5 min | ~7 min |
| Attribute verification (6-7 questions) | ~12s | ~12s |
| **Total (no retries)** | **~2 min** | **~8 min** |

AOI image-edit runs (`config_image_edit_aoi.yaml`) add ~30–40s per sample for the alignment post-processor (single-level GPU MI search at the generator's output resolution).
