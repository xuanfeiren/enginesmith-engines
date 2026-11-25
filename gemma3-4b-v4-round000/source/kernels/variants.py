"""The NEFF variants the engine uses, their kernel arguments and shapes (build time and tests)."""
import numpy as np

D, KT, HD, S = 2560, 20, 256, 8192
NTOK = S + 512
VOCAB = 262208
TPRE = 256                      # prefill chunk (tokens per layer execution)
DEC_LOCAL_B = 1024              # sliding window
DEC_MULTI = [2, 4]
DEC_MULTI_MAXB = 8192   # 2-step NEFFs for every bucket (the persistent global K/V buffer made them slow beyond 1536)
DEC_GLOBAL_B = [512, 1024, 1536, 2048, 3072, 4096, 4608, 6144, 8192]
PRE_LOCAL_B = [256, 512, 1024, 1280]
PRE_GLOBAL_B = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192]

LAYER_ARGS = ["h", "pos", "wqkv", "wo", "wgu", "wd", "lnw", "qkw", "kc", "vc", "cos", "sin"]


def layer_shapes(T):
    return {
        "h": ((D, T), "bf16"), "pos": ((1, 1), "i32"),
        "wqkv": ((2, 128, 16 * KT * 128), "bf16"), "wo": ((2, 128, KT * 8 * 128), "bf16"),
        "wgu": ((2, 128, 80 * KT * 128), "bf16"), "wd": ((2, 128, KT * 40 * 128), "bf16"),
        "lnw": ((4, D), "bf16"), "qkw": ((2, HD), "bf16"),
        "kc": ((4, HD, S), "bf16"), "vc": ((S, 1024), "bf16"),
        "cos": ((128, S), "bf16"), "sin": ((128, S), "bf16"),
    }


HEAD_ARGS = ["h", "col", "pos", "pnext", "lnf", "embT", "lmask", "lids", "toks"]


def head_shapes(T):
    return {
        "h": ((D, T), "bf16"), "col": ((1, 1), "i32"), "pos": ((1, 1), "i32"), "pnext": ((1, 1), "i32"),
        "lnf": ((1, D), "bf16"), "embT": ((2, 128, 1025 * KT * 128), "bf16"),
        "lmask": ((2, 128, 1025), "f32"), "lids": ((2, 128, 1025), "f32"), "toks": ((NTOK, 1), "i32"),
    }


EMB_ARGS = ["toks", "pos", "emb"]
EMB_SHAPES = {"toks": ((NTOK, 1), "i32"), "pos": ((1, 1), "i32"), "emb": ((VOCAB, D), "bf16")}


def all_variants():
    """name -> (args, shapes, call expression on the kernel module)."""
    v = {}
    for b in []:
        v[f"pre_L{b}"] = (LAYER_ARGS, layer_shapes(TPRE), f"layer_body({', '.join(LAYER_ARGS)}, {TPRE}, {b}, True)")
    for b in []:
        v[f"pre_G{b}"] = (LAYER_ARGS, layer_shapes(TPRE), f"layer_body({', '.join(LAYER_ARGS)}, {TPRE}, {b}, False)")
    for b in []:
        v[f"p2_L{b}"] = (LAYER_ARGS, layer_shapes(TPRE), f"pre2_body({', '.join(LAYER_ARGS)}, {TPRE}, {b}, True)")
    for b in []:
        v[f"p2_G{b}"] = (LAYER_ARGS, layer_shapes(TPRE), f"pre2_body({', '.join(LAYER_ARGS)}, {TPRE}, {b}, False)")
    # prefill groups: W = 12 layers (LLLLLG x 2, layers 0-11 and 12-23), T = layers 24-33 (LLLLLGLLLL)
    for bl, bg in PRE1_COMBOS:
        v[f"pgW{bl}_{bg}"] = pre_group_variant("LLLLLGLLLLLG", bl, bg)
        v[f"pgT{bl}_{bg}"] = pre_group_variant("LLLLLGLLLL", bl, bg)
        v[f"pgT{bl}_{bg}K"] = pre_group_variant("LLLLLGLLLL", bl, bg, kvlast=True)
        v[f"pgA{bl}_{bg}K"] = pre_group_variant("LLLLLG" * 5 + "LLLL", bl, bg, kvlast=True, embed=True)
        v[f"pgA{bl}_{bg}KH"] = pre_group_variant("LLLLLG" * 5 + "LLLL", bl, bg, kvlast=True, embed=True, head=True)
    # 512-token chunks: 12-layer groups while the global window is small; 6-layer groups beyond (the 12-layer
    # NEFF measured 4.7% slower than two 6-layer ones at a 4096-key global window)
    for t2, combos in P2S_COMBOS.items():
        for bl, bg in combos:
            if bg <= P2S_W_MAX:
                v[f"p2s{t2}W{bl}_{bg}"] = pre_group_variant("LLLLLGLLLLLG", bl, bg, t2)
                v[f"p2s{t2}T{bl}_{bg}"] = pre_group_variant("LLLLLGLLLL", bl, bg, t2)
                v[f"p2s{t2}T{bl}_{bg}K"] = pre_group_variant("LLLLLGLLLL", bl, bg, t2, kvlast=True)
                v[f"p2s{t2}A{bl}_{bg}K"] = pre_group_variant("LLLLLG" * 5 + "LLLL", bl, bg, t2, kvlast=True, embed=True)
                v[f"p2s{t2}A{bl}_{bg}KH"] = pre_group_variant("LLLLLG" * 5 + "LLLL", bl, bg, t2, kvlast=True, embed=True,
                                                              head=True)
            else:
                v[f"p2s{t2}M{bl}_{bg}"] = pre_group_variant("LLLLLG", bl, bg, t2)
                v[f"p2s{t2}A{bl}_{bg}K"] = pre_group_variant("LLLLLG" * 5 + "LLLL", bl, bg, t2, kvlast=True, embed=True)
                v[f"p2s{t2}A{bl}_{bg}KH"] = pre_group_variant("LLLLLG" * 5 + "LLLL", bl, bg, t2, kvlast=True, embed=True,
                                                              head=True)
        v[f"p2s{t2}T4_1280"] = pre_group_variant("LLLL", 1280, 0, t2)
        v[f"p2s{t2}T4_1280K"] = pre_group_variant("LLLL", 1280, 0, t2, kvlast=True)
    v["embed_pre512"] = (EMB_ARGS, EMB_SHAPES, "embed_pre_body(toks, pos, emb, 512)")
    v["head_pre512"] = (HEAD_ARGS, head_shapes(512), "head_body(" + ", ".join(HEAD_ARGS) + ", 512)")
    for kind, kinds in []:   # GROUPS: 3 NEFFs per decode step (replaced by dgA)
        if "G" in kinds:
            for b in DEC_GLOBAL_B:
                v[f"dg{kind}{b}"] = group_variant(kind, kinds, b)
        else:
            v[f"dg{kind}"] = group_variant(kind, kinds, 0)
    for b in DEC_GLOBAL_B:   # one NEFF per decode step: embedding, layers 0-33, LM head
        v[f"dgA{b}"] = group_variant("A", "LLLLLG" * 5 + "LLLL", b)
        for n in (DEC_MULTI if b <= DEC_MULTI_MAXB else []):   # n decode steps per NEFF
            a, sh, call = group_variant("A", "LLLLLG" * 5 + "LLLL", b)
            call = call.replace("dec_group_body(None, ", "dec_multi_body(")
            call = call.replace(", True, True, lnf,", f", {n}, lnf,")
            v[f"dgA{n}x{b}"] = (a, sh, call)
    for t in [TPRE, 512]:   # prompt's last token: layer 33 (decode arithmetic) + LM head + argmax
        v[f"hl33_{t}"] = last_variant(t)
    v["embed_pre"] = (EMB_ARGS, EMB_SHAPES, f"embed_pre_body(toks, pos, emb, {TPRE})")
    v["head_pre"] = (HEAD_ARGS, head_shapes(TPRE), f"head_body({', '.join(HEAD_ARGS)}, {TPRE})")
    return v


LW = ["wqkv", "wo", "wgu", "wd", "lnw", "qkw", "kc", "vc"]
GROUPS = [("F", "LLLLLGLLLLLG"), ("M", "LLLLLGLLLLLG"), ("T", "LLLLLGLLLL")]   # decode: layers 0-11 (+ embedding), 12-23, 24-33 (+ LM head)   # first (with embedding), middle, tail (with LM head)


def last_variant(T):
    args, shapes, call = group_variant("T", "L", 0)
    args = ["col"] + args
    shapes["h"] = ((D, T), "bf16")
    shapes["col"] = ((1, 1), "i32")
    call = call[:-1] + f", col, {T})"
    return args, shapes, call


def group_variant(kind, kinds, B):
    shp = layer_shapes(1)
    args, shapes = [], {}
    first, last = kind in ("F", "A"), kind in ("T", "A")
    if first:
        args += ["toks", "emb"]
        shapes.update({"toks": ((NTOK, 1), "i32"), "emb": ((VOCAB, D), "bf16")})
    else:
        args += ["h"]
        shapes["h"] = ((D, 1), "bf16")
    args += ["pos"]
    shapes["pos"] = ((1, 1), "i32")
    if last:
        args += [a for a in ["toks", "pnext", "lnf", "embT", "lmask", "lids"] if a not in args]
        hs = head_shapes(1)
        for a in ["toks", "pnext", "lnf", "embT", "lmask", "lids"]:
            shapes[a] = hs[a]
    rope = []
    if "L" in kinds:
        rope += ["cosl", "sinl"]
    if "G" in kinds:
        rope += ["cosg", "sing"]
    for a in rope:
        args.append(a)
        shapes[a] = ((128, S), "bf16")
    lws = []
    for i in range(len(kinds)):
        names = [f"l{i}_{n}" for n in LW]
        args += names
        for n, a in zip(LW, names):
            shapes[a] = shp[n]
        lws.append("[" + ", ".join(names) + "]")
    g = lambda a: a if a in args else "None"
    call = (f"dec_group_body({g('h')}, {g('toks')}, {g('emb')}, pos, {g('pnext')}, {g('cosl')}, {g('sinl')}, "
            f"{g('cosg')}, {g('sing')}, [{', '.join(lws)}], '{kinds}', {B}, {first}, {last}, {g('lnf')}, "
            f"{g('embT')}, {g('lmask')}, {g('lids')})")
    return args, shapes, call


P2S_W_MAX = 2048            # largest global bucket with 12-layer prefill groups
PRE1_COMBOS = [(256, 256)]   # single 256-token chunk (prompts <= 256)
PRE_COMBOS = [(256, 256), (512, 512), (1024, 1024), (1280, 2048), (1280, 3072), (1280, 4096), (1280, 6144),
              (1280, 8192)]


def pre_group_variant(kinds, BL, BG, T2=None, fn3=False, kvlast=False, embed=False, head=False):
    TT = TPRE if T2 is None else T2
    shp = layer_shapes(TPRE)
    if embed:   # the chunk's embedding inside the NEFF (instead of h from embed_pre)
        args = ["toks", "emb", "pos"]
        shapes = {"toks": ((NTOK, 1), "i32"), "emb": ((VOCAB, D), "bf16"), "pos": ((1, 1), "i32")}
    else:
        args = ["h", "pos"]
        shapes = {"h": ((D, TT), "bf16"), "pos": ((1, 1), "i32")}
    rope = ["cosl", "sinl"] + (["cosg", "sing"] if "G" in kinds else [])
    for a in rope:
        args.append(a)
        shapes[a] = ((128, S), "bf16")
    lws = []
    for i in range(len(kinds)):
        names = [f"l{i}_{n}" for n in LW]
        args += names
        for n, a in zip(LW, names):
            shapes[a] = shp[n]
        lws.append("[" + ", ".join(names) + "]")
    if head:   # last chunk: + the last token's layer 33, LM head and argmax (pre2s_group_body's head)
        hs = head_shapes(1)
        for a in ["col", "tpos", "pnext", "lnf", "embT", "lmask", "lids"]:
            args.append(a)
            shapes[a] = ((1, 1), "i32") if a in ("col", "tpos") else hs[a]
    g = lambda a: a if a in args else "None"
    fn = "pre1s_group_body" if T2 is None else ("pre3_group_body" if fn3 else "pre2s_group_body")
    call = (f"{fn}({'None' if embed else 'h'}, pos, cosl, sinl, {g('cosg')}, {g('sing')}, [{', '.join(lws)}], '{kinds}', {BL}, {BG}, "
            f"{TT}" + (", True" if kvlast else ", False") + (", toks, emb" if embed else "") +
            (", head=[col, tpos, pnext, lnf, embT, lmask, lids])" if head else ")"))
    return args, shapes, call


P2S_COMBOS = {512: [(512, 512), (1024, 1024), (1280, 1536), (1280, 2048), (1280, 2560), (1280, 3072), (1280, 3584),
                    (1280, 4096), (1280, 6144),
                                      (1280, 8192)]}


def used_variants():
    """The NEFFs the engine loads (all_variants also has the older group variants, kept for experiments)."""
    v = all_variants()
    keep = lambda n: (n in ("pgA256_256K", "pgA256_256KH", "hl33_256", "hl33_512") or
                      (n.startswith("p2s512A") and (n.endswith("K") or n.endswith("KH")))
                      or n.startswith("dgA"))
    return [n for n in v if keep(n)]


def zeros_for(shapes):
    import ml_dtypes
    dt = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "i32": np.int32}
    return {k: np.zeros(s, dt[t]) for k, (s, t) in shapes.items()}
