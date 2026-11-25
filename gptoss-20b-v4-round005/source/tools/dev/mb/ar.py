# micro-benchmark: N dependent all-reduces of [120, NC] fp32 over 4 ranks x 2 physical cores. usage: ar.py MODE N NC
import sys
import numpy as np
import nki, nki.language as nl, nki.isa as nisa, nki.collectives as ncc
from nki.collectives import ReplicaGroup
MODE = sys.argv[1]; N = int(sys.argv[2]); NC = int(sys.argv[3]); P = 120
TAGS = ["_" + str(i) for i in range(64)]


@nki.jit
def k(x, rg):
    pid = nl.program_id(0)
    out = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.shared_hbm)
    cur = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cur, src=x)
    if MODE == "n":
        c2 = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=c2, data=cur, op0=nl.add, operand0=pid * 10.0)
        cur = c2
    for i in range(N):
        if MODE == "n":
            res = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
            ncc.all_reduce(srcs=[cur], dsts=[res], op=nl.add, replica_group=rg)
            nxt = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nxt, data=res, op0=nl.multiply, operand0=0.125)
            cur = nxt
            continue
        other = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=cur, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                      dma_engine=(nisa.dma_engine.gpsimd_dma if NC * 4 <= 1024 else nisa.dma_engine.dma))
        both = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=both, data1=cur, data2=other, op=nl.add)
        res = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
        if MODE == "h":
            src = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.shared_hbm, name="s" + TAGS[i])
            dst = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.shared_hbm, name="d" + TAGS[i])
            nisa.dma_copy(dst=src, src=both)
            nisa.core_barrier(data=src, cores=(0, 1))
            ncc.all_reduce(srcs=[src], dsts=[dst], op=nl.add, replica_group=rg)
            nisa.dma_copy(dst=res, src=dst)
        elif MODE == "g":
            gat = nl.ndarray((P, 4 * NC), dtype=nl.float32, buffer=nl.sbuf)
            ncc.all_gather(srcs=[both], dsts=[gat], replica_group=rg, collective_dim=1)
            g01 = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=g01, data1=gat[0:P, 0:NC], data2=gat[0:P, NC:2 * NC], op=nl.add)
            g23 = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=g23, data1=gat[0:P, 2 * NC:3 * NC], data2=gat[0:P, 3 * NC:4 * NC], op=nl.add)
            nisa.tensor_tensor(dst=res, data1=g01, data2=g23, op=nl.add)
        elif MODE == "x":  # sendrecv only
            nisa.tensor_copy(dst=res, src=both)
        else:
            ncc.all_reduce(srcs=[both], dsts=[res], op=nl.add, replica_group=rg)
        # scale down to keep values bounded: res / 8
        nxt = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nxt, data=res, op0=nl.multiply, operand0=0.125)
        cur = nxt
    out2 = nl.ndarray((2, P, NC), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out2[pid], src=cur)
    return out2


W = 4
rg = ReplicaGroup([list(range(W))])
rng = np.random.default_rng(1)
xs = [rng.standard_normal((P, NC)).astype(np.float32) for r in range(W)]
res = k[2](xs, rg=rg)
r0 = np.asarray(res[0] if isinstance(res, (list, tuple)) else res)
ex = 2 * sum(x.astype(np.float64) for x in xs) / 8
np.save("ar_%s.npy" % MODE, r0)
print("maxrel", float(np.abs(r0[0] - ex).max() / np.abs(ex).max()), "pid diff", float(np.abs(r0[0]-r0[1]).max()))
print("out", r0[0].flat[0], r0[1].flat[0], "expected", 2 * (1 + 2 + 3 + 4) / 8 if N == 1 else "chain (1 step: 2.5)")
