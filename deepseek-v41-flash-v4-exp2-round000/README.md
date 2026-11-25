# deepseek-v41-flash-v4-exp2-round000

A specialized inference engine for [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) at revision `314113f8a7dadb388215396ba5fbee2c610a04d6`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.4 LTS
- Reference: custom, correctness only (no speed baseline), tensor-parallel degree 64 (the engine gets as many accelerator cores)
- Judged under evaluation version 4
- Decode: one forward pass per generated token, no speculative decoding (a rule in the session brief; the harness cannot check it)
- Source and data: https://github.com/xuanfeiren/enginesmith-engines/tree/main/deepseek-v41-flash-v4-exp2-round000
- Search: 1 rounds by Claude Code 2.1.317 (Claude Code) (model not reported, reasoning effort not reported)

## Notes from the author

- No framework runs this model correctly on Trainium (the speed section lists what was tried), so the reference is a CPU implementation that only decides correctness, and the score is the fraction of the hardware bound (evaluation version 4), not a speedup.
- This run re-evaluated one engine under evaluation version 4, with no new coding sessions: round 008 of an earlier search (8 rounds by Claude Code with Claude Opus 5.5, agent cost $1,350.06), with one change made by hand afterwards: an exact `exp` in place of a lookup-table `exp` in the prefill and decode softmax. With the lookup table, the engine's tokens drifted further from the reference's, and in the held-out final one stream crossed the per-stream gate at a router near-tie; with the exact `exp` the held-out final passes at the same speed. That search is in `search-history.csv` and in "How the search went" below, where its scores are against the CPU reference's speed (evaluation version 3), not fractions of the bound.

## Speed (held-out final evaluation; no framework baseline)

The reference decides correctness only and is no speed baseline (frameworks tried: NxD Inference 0.10: DeepSeek-V3 only, no deepseek_v41. transformers: no deepseek_v41. vllm-neuron 0.24 and optimum-neuron 0.4.5: no DeepSeek-V4. The official inference kernels (TileLang) are CUDA-only. Reference: the official model.py and engram.py on the CPU with the six TileLang kernels replaced by plain torch code doing the same math (validated: 15 of 15 kernels, deterministic, 0 mismatches judging its own tokens).); the score is the weighted geometric mean of the fractions of the hardware bound, bound time / engine time (a nominal bound where it is not computed, so then only a relative measure). Score 0.021; prefill 0.002, decode 0.265. Medians over the paired rounds, 65 tokens per request. The bound is not computed here: prefill: a quantized checkpoint (its packed tensors do not give the parameter count).

| prompt tokens | prefill, ms | decode, ms per token | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 231.8 | 40.6 | not computed | 27.1% |
| 1,024 | 561.1 | 41.5 | not computed | 26.5% |
| 4,096 | 1,924.0 | 42.4 | not computed | 25.9% |

Correctness: passed; 119 of 2080 judged steps mismatch (at most 208 allowed).

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.00x as long as one request for the same tokens at 256 prompt tokens (2,588.5 ms against 2,575.4 ms; each request 39.02 to 48.35 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went (the earlier search)

- 8 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 4182 min.
- First correct engine: round 001, 742 min after the start.
- First engine as fast as the reference (score 1.0): round 001, 742 min after the start.
- Agent usage: 2,664 input and 1,439k output tokens, 349,479k cache reads, $1350.06 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 741.8 | 38.16x | 57.25x | 25.43x | - / - | 4,812 |
| 002 | optimize | accepted (score_improved) | 1,193.6 | 46.35x | 64.29x | 33.42x | 416 / 213k | 5,901 |
| 003 | optimize | rejected (failed) | 1,816.4 | - | 84.13x | 38.59x | 626 / 420k | 6,548 |
| 004 | optimize | rejected (unchanged) | 1,891.3 | - | - | - | - / - | 5,901 |
| 005 | optimize | accepted (score_improved) | 2,176.5 | 55.80x | 82.89x | 37.56x | 234 / 97k | 6,745 |
| 006 | optimize | rejected (failed) | 2,640.1 | - | - | - | 410 / 256k | 7,235 |
| 007 | optimize | accepted (score_improved) | 3,038.3 | 66.65x | 114.51x | 38.80x | 290 / 154k | 7,922 |
| 008 | optimize | accepted (score_improved) | 4,181.6 | 74.11x | 122.99x | 44.66x | 688 / 300k | 8,546 |

`search-history.csv` has every round of that search with every prompt length.

## This re-evaluation

- 0 coding sessions by Claude Code 2.1.317 (Claude Code) (model not reported, reasoning effort not reported); the search took 260 min.
- First correct engine: round 000, 260 min after the start.
- Agent usage: not reported by the agent.

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 000 | seed | accepted (first_passing_candidate) | 260.2 | 0.021 | 0.002 | 0.265 | - / - | - |

`history.csv` in https://github.com/xuanfeiren/enginesmith-engines/tree/main/deepseek-v41-flash-v4-exp2-round000 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local the reference):

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash --revision 314113f8a7dadb388215396ba5fbee2c610a04d6 --local-dir models/DeepSeek-V4.1-Flash
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-engines/tree/main/deepseek-v41-flash-v4-exp2-round000 --model models/DeepSeek-V4.1-Flash
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
