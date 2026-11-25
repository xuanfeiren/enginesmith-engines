# q32-round011

A specialized inference engine for [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) at revision `9216db5781bf21249d130ec9da846c4624c16137`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.4 LTS
- Reference: nxdi 0.10.18399+ed62453e, tensor-parallel degree 8 (the engine gets as many accelerator cores)
- Judged under evaluation version 3
- Decode: one forward pass per generated token, no speculative decoding (a rule in the session brief; the harness cannot check it)
- Source and data: https://github.com/xuanfeiren/enginesmith-engines/tree/main/q32-round011
- Search: 11 rounds by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium), agent cost $557.56

## Notes from the author

- The final evaluation below ran when an engine could still see 16 cores. A clean rebuild of round 011, checked again on fresh held-out prompts with the engine given only the reference's 8 cores: passed; score 1.65x NxD Inference (prefill 1.91x, decode 1.42x). Decode probe 1.06.

## Speed against nxdi (held-out final evaluation)

Score 1.66x; prefill 1.91x, decode 1.43x (reference time / engine time, 65 tokens per request; the fraction of the hardware bound is bound time / engine time, an estimate).

| prompt tokens | prefill | decode | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 2.00x | 1.39x | not computed | not computed |
| 1,024 | 2.32x | 1.49x | not computed | not computed |
| 4,096 | 1.51x | 1.42x | not computed | not computed |

Correctness: passed; 5 of 2080 judged steps mismatch (at most 208 allowed).

## Speed in absolute terms

Medians over the paired rounds of the held-out final evaluation, this engine and nxdi on the same machine.

| prompt tokens | time to first token, ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 256 | 33.7 (nxdi 67.8) | 7,586.2 (3,778.1) | 73.9 (53.1) |
| 1,024 | 94.3 (nxdi 219.1) | 10,857.4 (4,673.1) | 74.2 (49.8) |
| 4,096 | 345.6 (nxdi 522.2) | 11,850.8 (7,843.2) | 72.5 (50.9) |

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.05x as long as one request for the same tokens at 256 prompt tokens (913.4 ms against 866.0 ms; each request 13.96 to 14.60 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went

- 11 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 3108 min.
- First correct engine: round 001, 116 min after the start.
- First engine as fast as nxdi (score 1.0): round 002, 334 min after the start.
- Agent usage: 3,278 input and 1,641k output tokens, 372,383k cache reads, $557.56 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 116.4 | 0.96x | 0.73x | 1.27x | 86 / 28k | 6,414 |
| 002 | optimize | accepted (score_improved) | 334.1 | 1.41x | 1.54x | 1.30x | 150 / 73k | 6,657 |
| 003 | optimize | accepted (score_improved) | 597.2 | 1.46x | 1.54x | 1.38x | 268 / 110k | 6,698 |
| 004 | optimize | accepted (score_improved) | 980.8 | 1.49x | 1.59x | 1.40x | 476 / 265k | 7,395 |
| 005 | optimize | accepted (segment_improved) | 1,168.5 | 1.50x | 1.62x | 1.39x | 314 / 94k | 7,498 |
| 006 | optimize | accepted (score_improved) | 1,455.4 | 1.55x | 1.74x | 1.39x | 288 / 196k | 7,811 |
| 007 | optimize | accepted (score_improved) | 1,807.2 | 1.61x | 1.84x | 1.41x | 350 / 186k | 8,683 |
| 008 | optimize | rejected (insufficient_improvement) | 2,327.3 | 1.61x | 1.87x | 1.39x | 428 / 226k | 9,271 |
| 009 | optimize | accepted (score_improved) | 2,473.2 | 1.63x | 1.92x | 1.39x | 240 / 92k | 9,652 |
| 010 | optimize | rejected (insufficient_improvement) | 2,676.0 | 1.63x | 1.91x | 1.38x | 282 / 127k | 10,344 |
| 011 | optimize | accepted (segment_improved) | 3,107.7 | 1.65x | 1.91x | 1.42x | 396 / 244k | 11,200 |

`history.csv` in https://github.com/xuanfeiren/enginesmith-engines/tree/main/q32-round011 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local nxdi):

```sh
hf download Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --local-dir models/Qwen3-32B
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-engines/tree/main/q32-round011 --model models/Qwen3-32B
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
