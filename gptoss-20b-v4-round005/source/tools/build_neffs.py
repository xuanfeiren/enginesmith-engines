"""Build time only: compile the engine's NKI kernels to NEFFs through NKI's documented route (NKI_ARTIFACTS_DIR set
before nki is imported, then the @nki.jit kernel is called once with numpy arrays of the right shapes; it compiles,
runs once on the visible NeuronCores and leaves kernel.neff). Each variant runs in its own process. NEFFs are cached
under target/neffcache keyed by the kernel source and the variant.

usage: python build_neffs.py OUTDIR            (all variants)
       python build_neffs.py --one NAME         (internal: build one variant in this process)
"""
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KDIR = os.path.join(ROOT, "kernels")
MAXPOS = 4608
TOKLEN = MAXPOS + 8
FULL_BUCKETS = [384, 512, 1152, 2048, 4224, 4608]  # = src/engine.cpp FULL_BUCKETS
PF_BUCKETS = [512, 1024, 2048, 4608]
MOE_TILES = [8, 16, 24, 32, 36]  # (no NT = 4: with fp8 expert weights pmoe4 computed some tokens of its block loop wrong, NOTES.md); prefill MoE NEFFs: prompt positions [0, 128 * NT)
ATT_CHUNKS = [2, 4, 8, 12, 16, 18]   # prefill attention NEFFs: chunks [0, NCHK) of 256 positions
GROUP = int(os.environ.get("ES_GROUP", "24"))              # decoder layers per decode NEFF (NL / GROUP NEFF executions per decode step)
VARIANTS = (["head"] + [f"step{b}" for b in FULL_BUCKETS] +
            ["pembed", "pgather"] + [f"pattfa_{k}{n}" for k in "sf" for n in ATT_CHUNKS] + [f"pmoe{t}" for t in MOE_TILES] +
            ["player_s", "player_f512", "pdense"])  # one-chunk prompts: the dense layer (pdense: all layers in one NEFF)


def build_one(name):
    assert os.environ.get("NKI_ARTIFACTS_DIR")
    import ml_dtypes
    import numpy as np
    sys.path.insert(0, KDIR)
    import model as M
    from nki.collectives import ReplicaGroup
    bf16 = ml_dtypes.bfloat16
    f8 = ml_dtypes.float8_e5m2  # expert weights (kernels/model.py WDT)
    W = 4
    rg = ReplicaGroup([list(range(W))])

    def z(shape, dt=np.float32):
        return np.zeros(shape, dt)

    def rep(a):
        return [a.copy() for _ in range(W)]

    def run(k, *args, **kw):
        """Compile (and run once) kernel k on 2 cores per rank; with ES_DUMP=DIR (development only), instead save one
        rank's inputs as DIR/<name>.npy (pos = ES_POS) and print the neuron-explorer ifmap arguments."""
        d = os.environ.get("ES_DUMP")
        if not d:
            return k[2](*args, **kw)
        import inspect
        out = []
        for n, a in zip(inspect.signature(k.func).parameters, args):
            a = a[0] if isinstance(a, list) else a
            if n == "pos":
                a = np.array([[int(os.environ.get("ES_POS", "0"))]], np.int32)
            np.save(os.path.join(d, n + ".npy"), a)
            out += [n, os.path.join(d, n + ".npy")]
        print("IFMAP " + " ".join(out))

    if name == "embed":
        run(M.embed, z((1, TOKLEN), np.int32), z((1, 1), np.int32), z((M.VOCAB, M.H), bf16))
    elif name == "inc":
        run(M.inc, z((1, 1), np.int32))
    elif name == "head":
        NG = M.NVT // M.LM_G
        run(M.head, rep(z((1, M.H), bf16)), rep(z((1, 1), np.int32)), rep(z((1, TOKLEN), np.int32)),
                  rep(z((M.P, M.HT))), rep(z((2 * NG, 128 * M.LM_F), bf16)), rep(z((2, 128, M.NVT))),
                  rep(z((2, 128, M.NVT))), rg=rg)
    elif name.startswith("layer_"):
        sliding = name == "layer_s"
        B = 128 if sliding else int(name[len("layer_f"):])
        T = MAXPOS
        args = [z((1, M.H), bf16), z((1, 1), np.int32), z((M.P, M.HT)), z((2, M.P, M.QKV_F), bf16),
                z((2, M.D, M.QKV_T)), z((MAXPOS, 128)), z((M.D, M.D), bf16), z((2, M.NQ, 1)),
                z((2, M.D, T), bf16), z((2, T, M.D), bf16), z((2, M.D, T), bf16), z((2, T, M.D), bf16),
                z((2, 128, M.WO_F // 2), bf16), z((M.P, M.HT)),
                z((M.P, M.HT)), z((M.P, M.HT * M.E), bf16), z((1, M.E)),
                z((M.E, 2 * M.P * M.GU_F), f8), z((M.E, 2 * M.P * M.GU_T)), z((M.E, 2 * M.P * M.DN_F), f8),
                z((M.E, M.H))]
        run(M.decode_layer, *[rep(a) for a in args], rg=rg, sliding=sliding, B=B)
    elif name.startswith("step") or name.startswith("dprof"):
        # dprofG_B: development only, a G-layer decode NEFF without embedding and head, for neuron-explorer
        prof = name.startswith("dprof")
        B = int(name.split("_")[1]) if prof else int(name[len("step"):])
        G = int(name[len("dprof"):].split("_")[0]) if prof else GROUP
        T = MAXPOS
        NG = M.NVT // M.LM_G
        XG = min(G, M.XS)
        if B >= int(os.environ.get("ES_SHARE_MAX", "100000")):  # SBUF: see kernels/model.py SHARE_W
            M.SHARE_W = False
        args = [z((1, M.H), bf16), z((1, 1), np.int32), z((1, TOKLEN), np.int32), z((M.VOCAB, M.H), bf16),
                z((MAXPOS, 128)), z((M.D, M.D), bf16),
                z((G, M.P, M.HT)), z((G, 2, 128, M.QKV_F), bf16), z((G, 2, M.D, M.QKV_T)), z((G, 2, M.NQ, 1)),
                z((G, 2, M.D, T), bf16), z((G, 2, T, M.D), bf16), z((G, 2, M.D, T), bf16), z((G, 2, T, M.D), bf16),
                z((G, 2, 128, M.WO_F // 2), bf16), z((G, M.P, M.HT)),
                z((G, M.P, M.HT)), z((G, M.P, M.HT * M.E), bf16), z((G, 1, M.E)),
                z((XG * M.E, 2 * 128 * M.GU_F), f8), z((G * M.E, 2 * M.P * M.GU_T)),
                z((XG * M.E, 2 * 128 * M.DN_F), f8), z((G * M.E, M.H)),
                z((max(G - M.XS, 1) * M.E, 2 * 128 * M.GU_F), f8), z((max(G - M.XS, 1) * M.E, 2 * 128 * M.DN_F), f8),
                z((M.P, M.HT)), z((2 * NG, 128 * M.LM_F), bf16), z((2, 128, M.NVT)), z((2, 128, M.NVT))]
        run(M.decode_step, *[rep(a) for a in args], rg=rg, G=G, B=B, first=not prof, last=not prof)
    elif name == "pembed":
        import prefill as PF
        run(PF.prefill_embed, z((1, TOKLEN), np.int32), z((1, 1), np.int32), z((M.VOCAB, M.H), bf16), z((M.H, MAXPOS), bf16))
    elif name == "pgather":
        import prefill as PF
        run(PF.prefill_gather, z((M.H, MAXPOS), bf16), z((1, 1), np.int32))
    elif name.startswith("pattn_"):
        import prefill as PF
        sliding = name == "pattn_s"
        KB = 0 if sliding else int(name[len("pattn_f"):])
        args = [z((M.H, MAXPOS), bf16), z((1, 1), np.int32), z((M.P, M.HT)), z((2, M.P, M.QKV_F), bf16),
                z((2, M.D, M.QKV_T)), z((128, MAXPOS)), z((M.D, M.D), bf16), z((2, M.NQ, 1)),
                z((2, M.D, MAXPOS), bf16), z((2, MAXPOS, M.D), bf16), z((2, M.D, M.WO_F), bf16), z((M.P, M.HT)),
                z((M.P, M.HT)), z((M.P, M.HT * M.E), bf16), z((1, M.E)),
                z((2, 2, M.P, M.HT * PF.SC)), z((MAXPOS, M.H), bf16), z((MAXPOS, 8)), z((MAXPOS, M.E))]
        run(PF.prefill_attn, *[rep(a) for a in args], rg=rg, sliding=sliding, KB=KB)
    elif name.startswith("pattnm_"):
        import prefill as PF
        sliding = name[len("pattnm_")] == "s"
        NCHK = int(name[len("pattnm_s"):])
        args = [z((M.H, MAXPOS), bf16), z((M.P, M.HT)), z((2, M.P, M.QKV_F), bf16),
                z((2, M.D, M.QKV_T)), z((128, MAXPOS)), z((M.D, M.D), bf16), z((2, M.NQ, 1)),
                z((2, M.D, MAXPOS), bf16), z((2, MAXPOS, M.D), bf16), z((2, M.D, M.WO_F), bf16), z((M.P, M.HT)),
                z((M.P, M.HT)), z((M.P, M.HT * M.E), bf16), z((1, M.E)),
                z((2, 2, M.P, M.HT * PF.SC)), z((MAXPOS, M.H), bf16), z((MAXPOS, 8)), z((MAXPOS, M.E))]
        run(PF.prefill_attn_multi, *[rep(a) for a in args], rg=rg, sliding=sliding, NCHK=NCHK)
    elif name.startswith("pattfa_"):
        import prefill as PF
        sliding = name[len("pattfa_")] == "s"
        NCHK = int(name[len("pattfa_s"):])
        S = NCHK * PF.SC
        args = [z((M.H, MAXPOS), bf16), z((M.P, M.HT)), z((2, M.P, M.QKV_F), bf16),
                z((2, M.D, M.QKV_T)), z((128, MAXPOS)), z((M.D, M.D), bf16), z((2 * M.NQ, 1)),
                z((2, M.D, MAXPOS), bf16), z((2, MAXPOS, M.D), bf16), z((2, 128, M.WO_F // 2), bf16), z((M.P, M.HT)),
                z((M.P, M.HT)), z((M.P, M.HT * M.E), bf16), z((1, M.E)),
                z((2, MAXPOS, M.H)), z((MAXPOS, M.H), bf16), z((MAXPOS, 8)), z((MAXPOS, M.E)),
                z((2 * M.NQ, M.D, S), bf16), z((2, M.D, S), bf16), z((2, S, M.D), bf16), z((2 * M.NQ, M.D, S), bf16)]
        run(PF.prefill_attn_fa, *[rep(a) for a in args], rg=rg, sliding=sliding, NCHK=NCHK)
    elif name.startswith("pmoe"):
        import prefill as PF
        NT = int(name[len("pmoe"):])
        pl = np.array([[NT * 128]], np.int32)
        args = [z((M.H, MAXPOS), bf16), pl, z((MAXPOS, M.H), bf16), z((MAXPOS, 8)), z((MAXPOS, M.E)),
                z((M.E, 2 * M.P * M.GU_F), f8), z((M.E, 2 * M.GU_T * M.P)), z((M.E, 2 * M.P * M.DN_F), f8),
                z((M.E, M.H)), z((M.E * (MAXPOS + 512), 4)), z((4 * NT + 2, 4), np.int32), z((2 * 4 * MAXPOS, M.H), bf16), z((2, MAXPOS, M.H))]
        run(PF.prefill_moe, *[rep(a) for a in args], rg=rg, NT=NT)
    elif name == "pdense":
        import prefill as PF
        G = 24
        XG = M.XS
        args = [z((M.H, MAXPOS), bf16), z((1, 1), np.int32), z((G, M.P, M.HT)), z((G, 2, M.P, M.QKV_F), bf16),
                z((G, 2, M.D, M.QKV_T)), z((128, MAXPOS)), z((M.D, M.D), bf16), z((G, 2, M.NQ, 1)),
                z((G, 2, M.D, MAXPOS), bf16), z((G, 2, MAXPOS, M.D), bf16), z((G, 2, 128, M.WO_F // 2), bf16), z((G, M.P, M.HT)),
                z((G, M.P, M.HT)), z((G, M.P, M.HT * M.E), bf16), z((G, 1, M.E)),
                z((XG * M.E, 2 * M.P * M.GU_F), f8), z((G * M.E, 2 * M.P * M.GU_T)), z((XG * M.E, 2 * M.P * M.DN_F), f8),
                z((G * M.E, M.H)), z(((G - XG) * M.E, 2 * M.P * M.GU_F), f8), z(((G - XG) * M.E, 2 * M.P * M.DN_F), f8),
                z((G * M.E, 2 * M.GU_T * M.P)), z((2, 2, M.P, M.HT * PF.SC))]
        run(PF.prefill_dense, *[rep(a) for a in args], rg=rg, G=G, KB=512)
    elif name.startswith("player_"):
        import prefill as PF
        sliding = name == "player_s"
        KB = 0 if sliding else int(name[len("player_f"):])
        args = [z((M.H, MAXPOS), bf16), z((1, 1), np.int32), z((M.P, M.HT)), z((2, M.P, M.QKV_F), bf16),
                z((2, M.D, M.QKV_T)), z((128, MAXPOS)), z((M.D, M.D), bf16), z((2, M.NQ, 1)),
                z((2, M.D, MAXPOS), bf16), z((2, MAXPOS, M.D), bf16), z((2, 128, M.WO_F // 2), bf16), z((M.P, M.HT)),
                z((M.P, M.HT)), z((M.P, M.HT * M.E), bf16), z((1, M.E)),
                z((M.E, 2 * M.P * M.GU_F), f8), z((M.E, 2 * M.P * M.GU_T)), z((M.E, 2 * M.P * M.DN_F), f8),
                z((M.E, M.H)), z((M.E, 2 * M.GU_T * M.P)), z((2, 2, M.P, M.HT * PF.SC))]
        run(PF.prefill_layer, *[rep(a) for a in args], rg=rg, sliding=sliding, KB=KB)
    else:
        raise SystemExit("unknown variant " + name)


def src_hash():
    h = hashlib.sha256()
    for f in sorted(os.listdir(KDIR)):
        if f.endswith(".py"):
            h.update(open(os.path.join(KDIR, f), "rb").read())
    h.update(open(os.path.abspath(__file__), "rb").read())
    return h.hexdigest()[:16]


def core_groups():
    """Groups of 4 logical NeuronCores for parallel builds (each variant runs once on a 4-rank group), from
    NEURON_RT_VISIBLE_CORES ("a-b" or a single core)."""
    vc = os.environ.get("NEURON_RT_VISIBLE_CORES", "0-3")
    a, b = (int(x) for x in vc.split("-")) if "-" in vc else (int(vc), int(vc))
    n = (b - a + 1) // 4
    return [f"{a + 4 * i}-{a + 4 * i + 3}" for i in range(max(n, 1))]


def main(outdir):
    import concurrent.futures
    import threading
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(ROOT, "target", "neffcache")
    os.makedirs(cache, exist_ok=True)
    key = src_hash()
    keep = os.environ.get("ES_KEEP", "")  # development only: reuse outdir's NEFFs of these variant prefixes
    todo = []
    for name in VARIANTS:
        if keep and any(name.startswith(k) for k in keep.split(",")) and os.path.exists(os.path.join(outdir, f"{name}.neff")):
            print(f"[build_neffs] {name}: kept (ES_KEEP)", flush=True)
            continue
        todo.append(name)
    groups = core_groups()
    free = list(groups)
    lock = threading.Lock()
    port0 = int(os.environ.get("NEURON_RT_ROOT_COMM_ID", "localhost:61000").split(":")[1])

    def build(name):
        cached = os.path.join(cache, f"{name}-{key}.neff")
        if not os.path.exists(cached):
            with lock:
                g = free.pop()
            try:
                art = os.path.join(ROOT, "target", "build", name)
                shutil.rmtree(art, ignore_errors=True)
                os.makedirs(art)
                gi = groups.index(g)
                env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_RT_VISIBLE_CORES": g,
                       "NEURON_RT_ROOT_COMM_ID": f"localhost:{port0 + 50 * gi}"}
                print(f"[build_neffs] compiling {name} on cores {g}", flush=True)
                subprocess.run([sys.executable, os.path.abspath(__file__), "--one", name], check=True, env=env,
                               stdout=sys.stderr)
                shutil.copy(os.path.join(art, "kernel.neff"), cached + ".tmp")
                os.replace(cached + ".tmp", cached)
            finally:
                with lock:
                    free.append(g)
        shutil.copy(cached, os.path.join(outdir, f"{name}.neff"))
        print(f"[build_neffs] {name}: {os.path.getsize(cached)} bytes", flush=True)

    with concurrent.futures.ThreadPoolExecutor(len(groups)) as ex:
        for f in [ex.submit(build, n) for n in todo]:
            f.result()


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        build_one(sys.argv[2])
    else:
        main(sys.argv[1])
