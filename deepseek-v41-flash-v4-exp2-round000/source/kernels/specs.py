"""The NEFFs of the engine: kernel function and the shape/dtype of every argument (build time only).

The C++ host binds tensors by argument name (outputs are output_0, output_1, ... in return order) and reads the
sizes from the NEFF, so this file is the single source of truth for shapes.
"""
import numpy as np
import ml_dtypes

BF = ml_dtypes.bfloat16
F8 = ml_dtypes.float8_e4m3          # legacy fp8 (the TensorE format); routed experts re-encoded exactly
F32 = np.float32
I32 = np.int32
U8 = np.uint8

D = 5120
MAXPOS = 8192
CAP2 = MAXPOS // 2
CAP1 = MAXPOS


def tiled(N, K):
    return (128, (N // 128) * (K // 128) * 128)


ATTN_IN = [
    ('h', (4, D), BF), ('premix', (1, 4), F32),
    ('fnT', (128, 160 * 24), F32), ('hsc', (1, 3), F32), ('hbs', (1, 24), F32),
    ('anw', (40, 128), BF), ('wqa', tiled(1280, D), BF), ('qnw', (10, 128), BF), ('wqb', tiled(32768, 1280), BF),
    ('wkv', tiled(512, D), BF), ('kvw', (4, 128), BF), ('rotP', (128, 128), BF),
    ('ropec', (128, 2), F32), ('roper', (1, 64), F32),
]

ATTN_OUT_COMMON_HEAD = [('h', (4, D), BF), ('mix', (1, 24), F32), ('posv', (1, 8), I32), ('qc', (128, 256), BF),
                        ('win', (128, 512), BF), ('kvr', (4, 128), BF), ('sink', (1, 64), F32),
                        ('roper', (1, 64), F32)]
ATTN_OUT_TAIL = [('woa', (128, 8 * 8 * 32 * 128), BF), ('wob', tiled(D, 8192), BF), ('fnT', (128, 160 * 24), F32),
                 ('hsc', (1, 3), F32), ('hbs', (1, 24), F32), ('fnw', (40, 128), BF), ('wgT', (128, 40 * 384), BF),
                 ('gbias', (1, 384), F32)]


def attn_out_c(cap, T):
    return ATTN_OUT_COMMON_HEAD + [('ckv', (cap, 512), BF), ('cmask', (1, T), F32)] + ATTN_OUT_TAIL


# one routed expert = 2 tensors: we (w1 | w3 | w2, fp8 tiled) and se (s1 | s3 | s2, bf16 per partition)
WE = (3 * 2304 * D,)
SE = (3 * 2304 * 160,)
MOE = [
    ('x', (40, 128), BF), ('h2', (4, D), BF), ('fmix', (1, 24), F32), ('rw', (1, 8), F32), ('bm', (128, 4), BF)]
MOE += [(f'e_{j}', WE, F8) for j in range(6)] + [(f'c_{j}', SE, BF) for j in range(6)]
MOE += [('ws1', tiled(2304, D), BF), ('ws3', tiled(2304, D), BF), ('ws2', tiled(D, 2304), BF)]

COMPRESS1 = [('xn', (40, 128), BF), ('wkv', tiled(512, D), BF), ('cnw', (4, 128), BF),
             ('wk', tiled(128, 512), BF), ('knw', (1, 128), BF), ('roperc', (1, 64), F32)]
COMPRESS2_STORE = [('xn', (40, 128), BF), ('wkv', tiled(512, D), BF), ('wgate', tiled(512, D), BF),
                   ('state', (2, 128, 4), F32)]
COMPRESS2_EMIT = [('xn', (40, 128), BF), ('wkv', tiled(512, D), BF), ('wgate', tiled(512, D), BF),
                  ('state', (2, 128, 4), F32), ('cnw', (4, 128), BF), ('wk', tiled(128, 512), BF),
                  ('knw', (1, 128), BF), ('roperc', (1, 64), F32)]


def indexer(cap):
    return [('qr', (10, 128), BF), ('xn', (40, 128), BF), ('posv', (1, 8), I32), ('wqb', tiled(4096, 1280), BF),
            ('wpT', (128, 40 * 32), BF), ('rotP', (128, 128), BF), ('ropec', (128, 2), F32),
            ('kc', (cap, 128), BF), ('lstrict', (128, 128), F32)]


ENGRAM = [('h', (4, D), BF), ('rows', (24, 256), U8), ('rscale', (24, 8), U8), ('wkv', tiled(25600, 6144), BF),
          ('qw', (4, D), BF), ('kw', (4, D), BF)]
HEAD = [('h', (4, D), BF), ('premix', (1, 4), F32), ('nw', (40, 128), BF), ('wt', tiled(129280, D), BF)]

# name -> (module, function, args)
_OLD = {
    'attn_in': ('k_attn_in', 'attn_in', ATTN_IN),
    'attn_out_w': ('k_attn_out', 'attn_out_w', ATTN_OUT_COMMON_HEAD + ATTN_OUT_TAIL),

    'moe': ('k_moe', 'moe', MOE),
    'compress1': ('k_compress', 'compress1', COMPRESS1),
    'compress2_store': ('k_compress', 'compress2_store', COMPRESS2_STORE),
    'compress2_emit': ('k_compress', 'compress2_emit', COMPRESS2_EMIT),
    'engram': ('k_misc', 'engram', ENGRAM),
    'head': ('k_misc', 'head', HEAD),
}
NEFFS = {k: _OLD[k] for k in ('moe', 'engram', 'head')}
_E1 = [('x', (40, 128), BF), ('rw', (1, 8), F32), ('bm', (128, 4), BF), ('we', WE, F8), ('se', SE, BF)]
NEFFS['expert1'] = ('k_moe', 'expert1', _E1)
NEFFS['head_part'] = ('k_misc', 'head_part', [('h', (4, D), BF), ('premix', (1, 4), F32), ('nw', (40, 128), BF),
                                              ('wt', tiled(25856, D), BF)])
NEFFS['engram_kv'] = ('k_misc', 'engram_kv', [('rows', (24, 256), U8), ('rscale', (24, 8), U8), ('wkv', tiled(25600, 6144), BF)])
NEFFS['engram4'] = ('k_misc', 'engram4', [('h', (4, 4, D), BF), ('rows', (4, 24, 256), U8), ('rscale', (4, 24, 8), U8),
                                          ('wkv', tiled(25600, 6144), BF), ('qw', (4, D), BF), ('kw', (4, D), BF)])
NEFFS['engram_gate'] = ('k_misc', 'engram_gate', [('h', (4, D), BF), ('kvh', (128, 200), BF), ('qw', (4, D), BF), ('kw', (4, D), BF)])
NEFFS['head_final'] = ('k_misc', 'head_final', [('c5', (1, 10), F32)])
_FMIXIN = [('h2', (4, D), BF), ('fnTf', (128, 160 * 24), F32), ('hscf', (1, 3), F32), ('hbsf', (1, 24), F32)]
NEFFS['shared1'] = ('k_moe', 'shared1', [('x', (40, 128), BF), ('ws1', tiled(2304, D), BF), ('ws3', tiled(2304, D), BF),
                                          ('ws2', tiled(D, 2304), BF)] + _FMIXIN)
NEFFS['moe'] = ('k_moe', 'moe', [a for a in MOE if a[0] != 'fmix'] + _FMIXIN[1:])
NEFFS['combine'] = ('k_moe', 'combine', [('y6', (6, 128, 40), BF), ('s2', (128, 40), BF), ('h2', (4, D), BF),
                                          ('fmix', (1, 24), F32)])
_UNUSED = {}
for _t in (512, 1024, 2048, 4096):
    _UNUSED[f'attn_out_c2_{_t}'] = ('k_attn_out', 'attn_out_c', attn_out_c(CAP2, _t))
for _t in (512, 1024, 2048, 4096, 8192):
    _UNUSED[f'attn_out_c1_{_t}'] = ('k_attn_out', 'attn_out_c', attn_out_c(CAP1, _t))
for _cap in (1024, 2048, 4096):
    _UNUSED[f'indexer2_{_cap}'] = ('k_indexer', f'indexer_{_cap}', indexer(CAP2))
for _cap in (1024, 2048, 4096, 8192):
    _UNUSED[f'indexer1_{_cap}'] = ('k_indexer', f'indexer_{_cap}', indexer(CAP1))

# ---------------------------------------------------------------------------------- fused attention-half variants
import gen_layer_variants as _G
import os as _os
_G.generate(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'k_layer_variants.py'))


def _layer_shapes(ratio, T):
    cap = CAP2 if ratio == 2 else CAP1
    return {
        'h': ((4, D), BF), 'premix': ((1, 4), F32), 'posv': ((1, 8), I32), 'fnT': ((128, 160 * 24), F32),
        'hsc': ((1, 3), F32), 'hbs': ((1, 24), F32), 'anw': ((40, 128), BF), 'wqa': (tiled(1280, D), BF),
        'qnw': ((10, 128), BF), 'wqb': (tiled(32768, 1280), BF), 'wkv': (tiled(512, D), BF), 'kvw': ((4, 128), BF),
        'rotP': ((128, 128), BF), 'ropec': ((128, 2), F32), 'roper': ((1, 64), F32), 'win': ((128, 512), BF),
        'sink': ((1, 64), F32), 'woa': ((128, 8 * 8 * 32 * 128), BF), 'wob': (tiled(D, 8192), BF),
        'fnTf': ((128, 160 * 24), F32), 'hscf': ((1, 3), F32), 'hbsf': ((1, 24), F32), 'fnw': ((40, 128), BF),
        'wgT': ((128, 40 * 384), BF), 'gbias': ((1, 384), F32), 'ckv': ((cap, 512), BF), 'cmask': ((1, T), F32),
        'cwkv': (tiled(512, D), BF), 'cwg': (tiled(512, D), BF), 'state': ((2, 128, 4), F32), 'cnw': ((4, 128), BF),
        'wk': (tiled(128, 512), BF), 'knw': ((1, 128), BF), 'roperc': ((1, 64), F32),
        'iwqb': (tiled(4096, 1280), BF), 'iwp': ((128, 40 * 32), BF), 'kc': ((cap, 128), BF),
        'lstrict': ((128, 128), F32), 'wrank': ((1, 128), F32), 'lvalid': ((1, 128 if ratio == 0 else 640), F32),
        'ws1': (tiled(2304, D), BF), 'ws3': (tiled(2304, D), BF), 'ws2': (tiled(D, 2304), BF)}


for _name, (_ratio, _T, _comp, _index) in _G.variants().items():
    _sh = _layer_shapes(_ratio, _T)
    NEFFS[_name] = ('k_layer_variants', _name, [(a, _sh[a][0], _sh[a][1]) for a in _G.args_of(_ratio, _T, _comp, _index)])

for _name, (_ratio, _T, _comp, _index) in _G.variants_p().items():
    _sh = dict(_layer_shapes(_ratio, _T))
    _sh.update({'y6': ((6, 128, 40), BF), 's2': ((128, 40), BF), 'h2p': ((4, D), BF), 'fmixp': ((1, 24), F32)})
    NEFFS[_name] = ('k_layer_variants', _name, [(a, _sh[a][0], _sh[a][1]) for a in _G.args_of(_ratio, _T, _comp, _index, True)])

# 4-token prefill variants: per-token inputs stacked on a leading dimension of 4
_PER_TOKEN = {'h', 'premix', 'posv', 'ropec', 'roper', 'roperc', 'cmask', 'wrank', 'lvalid'}
for _name, (_ratio, _T, _comp, _index) in _G.variants_m().items():
    _sh = _layer_shapes(_ratio, _T)
    _args = []
    for a in _G.args_m(_ratio, _T, _comp, _index):
        shp, dt = _sh[a]
        if a in _PER_TOKEN:
            shp = (4,) + (shp[1:] if a in ('premix', 'posv', 'cmask', 'wrank', 'lvalid') else shp)
        _args.append((a, shp, dt))
    NEFFS[_name] = ('k_layer_variants', _name, _args)
import os as _os2
if _os2.environ.get('DSV4_LAB8'):
    for _name, (_ratio, _T, _comp, _index) in _G.variants_m8().items():
        _sh = _layer_shapes(_ratio, _T)
        _args = []
        for a in _G.args_of(_ratio, _T, _comp, _index):
            shp, dt = _sh[a]
            if a in _PER_TOKEN:
                shp = (8,) + (shp[1:] if a in ('premix', 'posv', 'cmask', 'wrank', 'lvalid') else shp)
            _args.append((a, shp, dt))
        NEFFS[_name] = ('k_layer_variants', _name, _args)
_M4 = [('x', (4, 40, 128), BF), ('h2', (4, 4, D), BF), ('rw', (4, 8), F32), ('bm', (128, 4), BF)]
_M4 += [(f'e_{m}_{j}', WE, F8) for m in range(4) for j in range(6)]
_M4 += [(f'c_{m}_{j}', SE, BF) for m in range(4) for j in range(6)]
_M4 += [('ws1', tiled(2304, D), BF), ('ws3', tiled(2304, D), BF), ('ws2', tiled(D, 2304), BF),
        ('fnTf', (128, 160 * 24), F32), ('hscf', (1, 3), F32), ('hbsf', (1, 24), F32)]
NEFFS['moe4'] = ('k_moe4', 'moe4', _M4)

# deduplicated M-token MoE variants (k_moeu): moeu{M}_{S}, S expert slots
import k_moeu as _KU


def moeu_variants():
    v = {f'moeu4_{S}': (4, S) for S in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24)}
    if _os2.environ.get('DSV4_LABU'):
        v['moeu8_26'] = (8, 26)
    return v


_KU.generate(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'k_moeu_gen.py'), moeu_variants())
for _name, (_M, _S) in moeu_variants().items():
    _a = [('x', (_M, 40, 128), BF), ('h2', (_M, 4, D), BF), ('rw', (_S, _M), F32), ('bm', (128, 4), BF)]
    _a += [(f'e_{s}', WE, F8) for s in range(_S)] + [(f'c_{s}', SE, BF) for s in range(_S)]
    _a += [('s2in', (_M, 128, 40), BF),      # the shared expert's output (computed by the attention half)
           ('fnTf', (128, 160 * 24), F32), ('hscf', (1, 3), F32), ('hbsf', (1, 24), F32)]
    NEFFS[_name] = ('k_moeu_gen', _name, _a)
