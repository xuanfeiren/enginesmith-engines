"""generated: the 4-token MoE NEFF (outputs: h_out, the ffn mixes); expert m*6+j as (e_m_j, c_m_j)"""
import nki
import k_moe as KM


@nki.jit
def moe4(x, h2, rw, bm, e_0_0, e_0_1, e_0_2, e_0_3, e_0_4, e_0_5, e_1_0, e_1_1, e_1_2, e_1_3, e_1_4, e_1_5, e_2_0, e_2_1, e_2_2, e_2_3, e_2_4, e_2_5, e_3_0, e_3_1, e_3_2, e_3_3, e_3_4, e_3_5, c_0_0, c_0_1, c_0_2, c_0_3, c_0_4, c_0_5, c_1_0, c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_2_0, c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_3_0, c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, ws1, ws3, ws2, fnTf, hscf, hbsf):
    L6 = [[], [], [], [], [], []]
    T0 = KM.expert_lists([e_0_0, e_0_1, e_0_2, e_0_3, e_0_4, e_0_5], [c_0_0, c_0_1, c_0_2, c_0_3, c_0_4, c_0_5])
    for k in range(6):
        L6[k].append(T0[k])
    T1 = KM.expert_lists([e_1_0, e_1_1, e_1_2, e_1_3, e_1_4, e_1_5], [c_1_0, c_1_1, c_1_2, c_1_3, c_1_4, c_1_5])
    for k in range(6):
        L6[k].append(T1[k])
    T2 = KM.expert_lists([e_2_0, e_2_1, e_2_2, e_2_3, e_2_4, e_2_5], [c_2_0, c_2_1, c_2_2, c_2_3, c_2_4, c_2_5])
    for k in range(6):
        L6[k].append(T2[k])
    T3 = KM.expert_lists([e_3_0, e_3_1, e_3_2, e_3_3, e_3_4, e_3_5], [c_3_0, c_3_1, c_3_2, c_3_3, c_3_4, c_3_5])
    for k in range(6):
        L6[k].append(T3[k])
    return KM.moe_impl_m(4, x, h2, None, rw, bm, L6[0], L6[1], L6[2], L6[3], L6[4], L6[5], ws1, ws3, ws2, fnTf, hscf, hbsf)
