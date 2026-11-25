# Third-party code

No third-party source is copied into this engine. The kernels (`kernels/gemma.py`, `kernels/pre2s.py`, `kernels/pre3.py`) are written against the public
NKI API (`nki.language`, `nki.isa`, `@nki.jit`, nki 0.6.0 / neuronx-cc 2.27.5334); the host (`host/`) uses only libnrt's
public headers. The layouts and the build route follow the EngineSmith skill's Trainium backend notes.

Build-time tools (not shipped in the executable): AWS Neuron SDK (nki, neuronx-cc) to compile the kernels to NEFFs,
and PyTorch (`kernels/gen_tables.py`) to compute the RoPE tables the same way transformers does.
