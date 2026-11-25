# enginesmith-engines

Specialized inference engines built with [EngineSmith](https://github.com/EngineSmith/enginesmith), one per directory. Each directory is a whole package: its README has the results and how to check them, `source/` the engine, `enginesmith.json` every setting and result, and `LICENSE` its license.

| engine | model | machine | score |
|---|---|---|---|
| [deepseek-v41-flash-v4-exp2-round000](deepseek-v41-flash-v4-exp2-round000/) | deepseek-ai/DeepSeek-V4.1-Flash | trn2.48xlarge | 0.021 of the hardware bound (held-out final) |
| [gemma3-4b-v4-round000](gemma3-4b-v4-round000/) | unsloth/gemma-3-4b-it | trn2.48xlarge | 0.800 of the hardware bound (held-out final) |
| [gptoss-20b-v4-round005](gptoss-20b-v4-round005/) | openai/gpt-oss-20b | trn2.48xlarge | 0.061 of the hardware bound (held-out final) |
| [llama8b-round011](llama8b-round011/) | NousResearch/Meta-Llama-3.1-8B-Instruct | trn2.48xlarge | 1.66x nxdi (search evaluation) |
| [q06-astra-y-round003](q06-astra-y-round003/) | Qwen/Qwen3-0.6B | trn2.48xlarge | 2.30x nxdi (search evaluation) |
| [q14-lab-round001](q14-lab-round001/) | Qwen/Qwen3-14B | trn2.48xlarge | 1.74x nxdi (search evaluation) |
| [q30-lab-round002](q30-lab-round002/) | Qwen/Qwen3-30B-A3B | trn2.48xlarge | 2.91x nxdi (search evaluation) |
| [q32-round011](q32-round011/) | Qwen/Qwen3-32B | trn2.48xlarge | 1.66x nxdi (held-out final) |
| [q35-2b-v4-round000](q35-2b-v4-round000/) | Qwen/Qwen3.5-2B | trn2.48xlarge | 0.071 of the hardware bound (held-out final) |
| [q8-opus-round003](q8-opus-round003/) | Qwen/Qwen3-8B | trn2.48xlarge | 1.58x nxdi (search evaluation) |

