# enginesmith-engines

Specialized inference engines built with [EngineSmith](https://github.com/EngineSmith/enginesmith), one per directory. Each directory is a whole package: its README has the results and how to check them, `source/` the engine, `enginesmith.json` every setting and result, and `LICENSE` its license.

| engine | model | machine | score |
|---|---|---|---|
| [deepseek-v41-flash-v4-exp2-round000](deepseek-v41-flash-v4-exp2-round000/) | deepseek-ai/DeepSeek-V4.1-Flash | trn2.48xlarge | 0.02x the reference (held-out final) |
| [gemma3-4b-v4-round000](gemma3-4b-v4-round000/) | unsloth/gemma-3-4b-it | trn2.48xlarge | 0.80x the reference (held-out final) |
| [gptoss-20b-v4-round005](gptoss-20b-v4-round005/) | openai/gpt-oss-20b | trn2.48xlarge | 0.06x the reference (held-out final) |
| [llama8b-round011](llama8b-round011/) | NousResearch/Meta-Llama-3.1-8B-Instruct | trn2.48xlarge | 1.66x nxdi (search evaluation) |

