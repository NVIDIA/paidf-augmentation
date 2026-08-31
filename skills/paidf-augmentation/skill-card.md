## Description: <br>
Use when authoring or validating PAIDF augmentation YAML configs, or running remote Cosmos Transfer/Predict, image-edit, or image-to-video inference. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to drive the PAIDF augmentation pipeline end to end — authoring and validating YAML configs, running remote generative AI inference (Cosmos Transfer, Cosmos Predict, image-edit, image-to-video), and configuring captioning and evaluation stages. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Configuration Schema](references/configuration-schema.md) <br>
- [Config Decision Tree](references/config-decision-tree.md) <br>
- [Pipeline Operations](references/pipeline-operations.md) <br>
- [Captioning Strategy Guide](references/captioning-strategy-guide.md) <br>
- [Evaluator Setup Guide](references/evaluator-setup-guide.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Image Attribute Augmentation](references/image-attribute-augmentation.md) <br>
- [Event Video Generation](references/event-video-gen.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 13 tasks (12 positive, 1 negative) from a curated evaluation dataset, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies routing quality and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 52% → 91% (+39 points) | 52% → 88% (+36 points) |
| Security | 92% → 100% (+8 points) | 73% → 92% (+19 points) |
| Correctness | 38% → 100% (+62 points) | 63% → 97% (+34 points) |
| Discoverability | 54% → 88% (+35 points) | 46% → 78% (+32 points) |
| Effectiveness | 36% → 87% (+51 points) | 39% → 89% (+49 points) |
| Efficiency | 42% → 82% (+41 points) | 40% → 86% (+46 points) |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
