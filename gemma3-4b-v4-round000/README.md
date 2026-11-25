# gemma3-4b-v4-round000

A specialized inference engine for [unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it) at revision `bf46152c47f5dd20b896357cb51abc4c03b8ee8c`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.4 LTS
- Reference: custom, correctness only (no speed baseline)
- Judged under evaluation version 4
- Decode: one forward pass per generated token, no speculative decoding (a rule in the session brief; the harness cannot check it)
- Source and data: https://github.com/xuanfeiren/enginesmith-engines/tree/main/gemma3-4b-v4-round000
- Search: 1 rounds by Claude Code 2.1.317 (Claude Code) (model not reported, reasoning effort not reported)

## Notes from the author

- No framework runs this model correctly on Trainium (the speed section lists what was tried), so the reference is a CPU implementation that only decides correctness, and the score is the fraction of the hardware bound (evaluation version 4), not a speedup.
- This run re-evaluated one engine under evaluation version 4, with no new coding sessions: round 008 of an earlier search (12 rounds by Claude Code with Claude Opus 5.5, agent cost $294.36). That search is in `search-history.csv` and in "How the search went" below, where its scores are against the CPU reference's speed (evaluation version 3), not fractions of the bound.

## Speed (held-out final evaluation; no framework baseline)

The reference decides correctness only and is no speed baseline (frameworks tried: NxD Inference 0.10: fails on the first forward pass on the device (RuntimeError in NxDModel.forward). vllm-neuron and optimum-neuron run through NxD Inference.); the score is the weighted geometric mean of the fractions of the hardware bound, bound time / engine time (a nominal bound where it is not computed, so then only a relative measure). Score 0.800; prefill 0.712, decode 0.900. Medians over the paired rounds, 65 tokens per request.

| prompt tokens | prefill, ms | decode, ms per token | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 15.0 | 12.0 | 65.9% | 89.9% |
| 1,024 | 52.5 | 12.0 | 76.7% | 90.8% |
| 4,096 | 229.8 | 12.2 | 71.3% | 89.2% |

Correctness: passed; 6 of 2080 judged steps mismatch (at most 208 allowed).

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.03x as long as one request for the same tokens at 256 prompt tokens (786.9 ms against 765.9 ms; each request 12.21 to 15.34 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went (the earlier search)

- 12 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 1858 min.
- First correct engine: round 001, 64 min after the start.
- First engine as fast as the reference (score 1.0): round 001, 64 min after the start.
- Agent usage: 2,506 input and 1,544k output tokens, 287,769k cache reads, $294.36 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 63.9 | 11.11x | 15.36x | 8.03x | 176 / 166k | 1,533 |
| 002 | optimize | accepted (score_improved) | 392.5 | 15.61x | 23.34x | 10.44x | 484 / 371k | 3,264 |
| 003 | optimize | accepted (case_improved) | 602.5 | 15.68x | 22.74x | 10.81x | 304 / 221k | 3,686 |
| 004 | optimize | rejected (insufficient_improvement) | 646.7 | 15.41x | 22.73x | 10.45x | 76 / 55k | 3,696 |
| 005 | optimize | accepted (score_improved) | 812.4 | 16.04x | 24.08x | 10.68x | 160 / 78k | 3,960 |
| 006 | optimize | rejected (insufficient_improvement) | 931.7 | 15.62x | 22.61x | 10.79x | 146 / 83k | 4,031 |
| 007 | optimize | rejected (insufficient_improvement) | 1,113.0 | 15.57x | 22.76x | 10.65x | 438 / 234k | 4,017 |
| 008 | optimize | accepted (score_improved) | 1,403.2 | 16.86x | 26.56x | 10.71x | 306 / 192k | 4,008 |
| 009 | optimize | rejected (insufficient_improvement) | 1,574.8 | 15.75x | 23.27x | 10.66x | 220 / 97k | 4,218 |
| 010 | optimize | rejected (insufficient_improvement) | 1,704.5 | 15.98x | 23.71x | 10.78x | 92 / 24k | 4,012 |
| 011 | optimize | rejected (insufficient_improvement) | 1,824.3 | 15.60x | 22.51x | 10.81x | 104 / 22k | 4,275 |
| 012 | optimize | rejected (stopped) | 1,857.7 | - | - | - | - / - | 4,275 |

`search-history.csv` has every round of that search with every prompt length.

## This re-evaluation

- 0 coding sessions by Claude Code 2.1.317 (Claude Code) (model not reported, reasoning effort not reported); the search took 46 min.
- First correct engine: round 000, 46 min after the start.
- Agent usage: not reported by the agent.

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 000 | seed | accepted (first_passing_candidate) | 45.5 | 0.802 | 0.715 | 0.900 | - / - | - |

`history.csv` in https://github.com/xuanfeiren/enginesmith-engines/tree/main/gemma3-4b-v4-round000 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local the reference):

```sh
hf download unsloth/gemma-3-4b-it --revision bf46152c47f5dd20b896357cb51abc4c03b8ee8c --local-dir models/gemma-3-4b-it
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-engines/tree/main/gemma3-4b-v4-round000 --model models/gemma-3-4b-it
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
