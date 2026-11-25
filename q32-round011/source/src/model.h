// Qwen3-32B shapes and the engine's compiled configuration (must match kernels/targets.py).
#pragma once
constexpr int W = 8;                     // tensor-parallel ranks
constexpr int C = 2 * W;                 // virtual cores
constexpr int L = 64, HID = 5120, HT = HID / 128, NQ = 64, NKV = 8, D = 128, INTER = 25600, VOCAB = 151936;
constexpr int IT = INTER / 128;          // 200 intermediate tiles
constexpr int NQL = NQ / C;              // 4 q heads per virtual core
constexpr int NKL = NKV >= C ? NKV / C : 1;   // 1: kv heads replicated (virtual core v uses kv head v * NKV / C)
constexpr int ITL = (IT + C - 1) / C;    // 13: tensor shapes keep ITL * C = 208 tiles per layer
// round 004 (HALF): virtual core v owns intermediate rows [IPC v, IPC v + IPC): ITF full tiles + one IH-row half tile
constexpr int IPC = INTER / C;           // 1600
constexpr int ITF = IPC / 128;           // 12
constexpr int IH = IPC - ITF * 128;      // 64
constexpr bool HALF = false;             // must match kernels/targets.py HALF (measured no faster: off)
constexpr int VT = (VOCAB + 127) / 128, NVL = (VT + C - 1) / C;
constexpr int TMAX = 4608;
constexpr int MAXPOS = TMAX;             // positions a request may reach
constexpr double ROPE_THETA = 1000000.0;
