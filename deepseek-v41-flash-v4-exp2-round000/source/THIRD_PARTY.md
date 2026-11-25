# Third-party code

No third-party source is copied into this engine.

- The NKI kernels are written against the public NKI API (`nki.language`, `nki.isa`) of the Neuron SDK
  (nki 0.6.0, neuronx-cc 2.27.5334).  Idioms (TensorE transposes through PSUM, HBM round trips for indices) were
  learned from AWS's NKI Library (nkilib, Apache-2.0, github.com/aws-neuron/nki-library) but no file was copied.
- `tools/gen_engram.py` imports the official DeepSeek-V4.1-Flash `inference/engram.py` (MIT) at generation time;
  its outputs `data/engram_map.bin` and `data/engram_consts.h` are derived data of the MIT-licensed checkpoint
  (tokenizer + hashing constants).
