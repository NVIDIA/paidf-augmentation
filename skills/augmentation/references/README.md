# Reference Files

Table of contents for the `augmentation` skill references.

| File | When to read |
|------|--------------|
| [configuration-schema.md](configuration-schema.md) | Full per-section YAML examples for `data`, `endpoints`, `pipeline`, `captioning`, `augmentation`, `data_processing`, and `evaluators`. Read when authoring or debugging a config. |
| [config-decision-tree.md](config-decision-tree.md) | Which config to start from, model selection, captioning strategy decisions, endpoint requirements, and the data_processing alignment override rules. |
| [captioning-strategy-guide.md](captioning-strategy-guide.md) | Deep-dive on all 5 captioning modes (VLM+LLM, VLM-only, LLM-only, text, file) with complete YAML examples and variable setup. |
| [evaluator-setup-guide.md](evaluator-setup-guide.md) | Hallucination check tuning, attribute verification with MCQ questions, extra-questions wiring, and natural-caption regeneration. |
| [troubleshooting.md](troubleshooting.md) | Config-validation errors, runtime errors (host vs. container), and typical per-stage inference timings. |
| [person-attribute-image-edit.md](person-attribute-image-edit.md) | PAS-specific image-edit workflow: generated-config compatibility fixes, single-ID grids, status reporting, augmented PAS dataset packaging. |
