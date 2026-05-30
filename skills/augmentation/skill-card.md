## Description: <br>
Run the NVIDIA Physical AI Data Factory (PAIDF) augmentation pipeline: author and validate YAML configs and run inference with Cosmos Transfer 2.5 (video style transfer), Cosmos Predict 2.5 (text/image/video-to-video), and image-edit models — covering captioning, control modalities, evaluators (hallucination, attribute, VLM verification), data_processing alignment, and PAS person-attribute edits via Docker GPU inference. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to augment camera data through NVIDIA generative AI models (Cosmos Transfer 2.5, Cosmos Predict 2.5, image edit) with automated captioning, generation, and quality evaluation for physical AI data pipelines. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Configuration Schema](references/configuration-schema.md) <br>
- [Config Decision Tree](references/config-decision-tree.md) <br>
- [Captioning Strategy Guide](references/captioning-strategy-guide.md) <br>
- [Evaluator Setup Guide](references/evaluator-setup-guide.md) <br>
- [Person Attribute Image Edit](references/person-attribute-image-edit.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML configuration] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>
- `codex` <br>



## Evaluation Tasks: <br>
Evaluated against 13 internal evaluation tasks (12 positive skill-activation, 1 negative) with 2 attempts per task and a 50% pass threshold. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 100% (+4%) | 100% (+0%) |
| Correctness | 8 | 60% (-3%) | 87% (-0%) |
| Discoverability | 8 | 20% (-5%) | 62% (-4%) |
| Effectiveness | 8 | 80% (-1%) | 93% (+3%) |
| Efficiency | 8 | 32% (-1%) | 53% (-3%) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
