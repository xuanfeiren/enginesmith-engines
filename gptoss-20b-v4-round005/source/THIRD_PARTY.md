# Third-party code

## AWS NKI Library (nkilib), Apache-2.0

`kernels/nkilib_attn/` is a verbatim copy of part of the AWS NKI Library (github.com/aws-neuron/nki-library), taken
from the copy validated against this compiler that ships inside the neuronx-cc wheel (neuronx-cc 2.27.5334,
`site-packages/nkilib/`). License: Apache License 2.0 (each file keeps its "Copyright Amazon.com, Inc. or its
affiliates" Apache-2.0 header). Files:

- `core/attention/attention_cte.py` (prefill flash attention; the engine calls `_attention_cte` from
  `kernels/prefill.py: prefill_attn_fa` with causal masking, sink tokens and, for sliding layers, a 128 window)
- `core/utils/allocator.py`, `common_types.py`, `interleave_copy.py`, `kernel_assert.py`, `kernel_helpers.py`,
  `logging.py`, `modular_allocator.py`, `stream_shuffle_broadcast.py`, `tree_logger.py` (its dependencies)
- empty `__init__.py` files were added to make the copy an importable package.

The kernel uses only the public NKI API.

## Everything else

The other NKI kernels (`kernels/model.py`, `kernels/prefill.py`), the build tools (`tools/`) and the host code
(`src/engine.cpp`) were written for this engine, using the public NKI API (`nki.language`, `nki.isa`,
`nki.collectives`) and the public libnrt headers (`/opt/aws/neuron/include/nrt/`).

Techniques follow the EngineSmith skill notes (column layout, weight-stationary GEMV, all-reduce pattern) and the
semantics of Hugging Face transformers' `modeling_gpt_oss.py` and `integrations/mxfp4.py` (Apache-2.0), which were
read, not copied.
