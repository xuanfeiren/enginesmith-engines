# gptoss-20b-v4-round005

A specialized inference engine for [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) at revision `6cee5e81ee83917806bbde320786a8fb61efebee`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.4 LTS
- Reference: custom, correctness only (no speed baseline), tensor-parallel degree 4 (the engine gets as many accelerator cores)
- Judged under evaluation version 4
- Decode: one forward pass per generated token, no speculative decoding (a rule in the session brief; the harness cannot check it)
- Source and data: https://github.com/xuanfeiren/enginesmith-engines/tree/main/gptoss-20b-v4-round005
- Search: 5 rounds by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium), agent cost $381.13

## Notes from the author

- No framework runs this model correctly on Trainium (the speed section lists what was tried), so the reference is a CPU implementation that only decides correctness, and the score is the fraction of the hardware bound (evaluation version 4), not a speedup.

## Speed (held-out final evaluation; no framework baseline)

The reference decides correctness only and is no speed baseline (frameworks tried: NxD Inference 0.10 (neuronx-distributed-inference 0.10.18399, tensor parallel 4): the reference process exits with status 1 during compile/load (es-wave3 prepare.log, 2026-10). vllm-neuron and optimum-neuron run through NxD Inference. PyTorch/XLA was not tried as a reference.); the score is the weighted geometric mean of the fractions of the hardware bound, bound time / engine time (a nominal bound where it is not computed, so then only a relative measure). Score 0.061; prefill 0.013, decode 0.279. Medians over the paired rounds, 65 tokens per request. The bound is not computed here: a mixture-of-experts model (the experts a token reads depend on the routing).

| prompt tokens | prefill, ms | decode, ms per token | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 30.4 | 3.5 | not computed | not computed |
| 1,024 | 70.2 | 3.6 | not computed | not computed |
| 4,096 | 208.3 | 3.7 | not computed | not computed |

Correctness: passed; 21 of 2080 judged steps mismatch (at most 208 allowed).

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.31x as long as one request for the same tokens at 256 prompt tokens (294.3 ms against 225.2 ms; each request 4.07 to 6.73 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went

- 5 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 1459 min.
- First correct engine: round 001, 161 min after the start.
- Agent usage: 2,202 input and 1,386k output tokens, 340,268k cache reads, $381.13 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 161.0 | 0.024 | 0.006 | 0.092 | 304 / 245k | 2,185 |
| 002 | optimize | accepted (score_improved) | 510.1 | 0.037 | 0.010 | 0.135 | 548 / 412k | 10,243 |
| 003 | optimize | accepted (score_improved) | 892.5 | 0.050 | 0.012 | 0.208 | 512 / 292k | 10,520 |
| 004 | optimize | accepted (score_improved) | 1,141.2 | 0.057 | 0.013 | 0.258 | 378 / 185k | 11,129 |
| 005 | optimize | accepted (score_improved) | 1,459.2 | 0.061 | 0.013 | 0.280 | 460 / 253k | 11,481 |

`history.csv` in https://github.com/xuanfeiren/enginesmith-engines/tree/main/gptoss-20b-v4-round005 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local the reference):

```sh
hf download openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee --local-dir models/gpt-oss-20b
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-engines/tree/main/gptoss-20b-v4-round005 --model models/gpt-oss-20b
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
