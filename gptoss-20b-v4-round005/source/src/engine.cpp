// gpt-oss-20b inference engine for Trainium2 (trn2, 4 logical NeuronCores, tensor parallel over 8 physical cores).
// Host code: loads the checkpoint (MXFP4 experts decoded exactly to bf16 at load), lays the weights out for the
// NKI kernels (kernels/model.py), embeds the NEFFs, and drives them through libnrt. The host never computes the
// model: every layer, the LM head and the argmax run on the NeuronCores.
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------------------------------------ constants
static const int H = 2880, P = 120, HT = 24, D = 64, NQ = 8, E = 32, NL = 24;
static const int QKV_T = 10, QKV_F = QKV_T * HT * D, WO_F = NQ * HT * P;
static const int GU_T = 6, GU_F = GU_T * HT * P, IT = 3, DN_F = IT * HT * P;
static const int VOCAB = 201088, NVT = 200, LM_G = 4, NG = NVT / LM_G, HT2 = 23, LM_F = LM_G * HT2 * 128;  // LM head: 128-row tiles
static const int MAXPOS = 4608, TOKLEN = MAXPOS + 8, SLIDE = 128;
static const int W = 4;  // tensor-parallel ranks (logical NeuronCores)
static const float BIG = 1048576.0f;
static const std::vector<int> FULL_BUCKETS = {384, 512, 1152, 2048, 4224, 4608};  // decode attention buckets (prompt + 128 above the timed 256 / 1024 / 4096)
static const std::vector<int> PF_BUCKETS = {512, 1024, 2048, 4608};  // prefill key buckets (>= chunk end)
static const int SC = 256, NCH = MAXPOS / SC;  // prefill chunk
static const std::vector<int> MOE_TILES = {8, 16, 24, 32, 36};  // no pmoe4 (NOTES.md)  // prefill MoE NEFFs: positions [0, 128 * NT)
static const std::vector<int> ATT_CHUNKS = {2, 4, 8, 12, 16, 18};  // prefill attention NEFFs: chunks [0, NCHK)

#define NEFF(n) extern "C" const char neff_##n##_start[], neff_##n##_end[];
NEFF(head) NEFF(step384) NEFF(step512) NEFF(step1152) NEFF(step2048) NEFF(step4224) NEFF(step4608)
NEFF(pembed) NEFF(pgather) NEFF(player_s) NEFF(player_f512) NEFF(pdense)
NEFF(pattfa_s2) NEFF(pattfa_s4) NEFF(pattfa_s8) NEFF(pattfa_s12) NEFF(pattfa_s16) NEFF(pattfa_s18)
NEFF(pattfa_f2) NEFF(pattfa_f4) NEFF(pattfa_f8) NEFF(pattfa_f12) NEFF(pattfa_f16) NEFF(pattfa_f18)
NEFF(pmoe8) NEFF(pmoe16) NEFF(pmoe24) NEFF(pmoe32) NEFF(pmoe36)

static FILE* g_out = nullptr;  // protocol replies (private copy of the original stdout)

static void die(const std::string& m) { fprintf(stderr, "[engine] fatal: %s\n", m.c_str()); fflush(stderr); exit(1); }
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { char b_[512]; \
  snprintf(b_, sizeof b_, "%s:%d %s -> %d (%s)", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); \
  throw std::runtime_error(b_); } } while (0)

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) {  // round to nearest even
  uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16);
}

// ------------------------------------------------------------------------------------------------ tiny JSON
struct J {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false; double n = 0; int64_t i = 0; bool isint = false; std::string s;
  std::vector<J> a; std::vector<std::pair<std::string, J>> o;
  const J* get(const std::string& k) const { for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
};
struct JP {
  const char* p; const char* e;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
  [[noreturn]] void bad() { throw std::runtime_error("bad json"); }
  std::string str() {
    if (*p != '"') bad(); p++; std::string r;
    while (p < e && *p != '"') {
      if (*p == '\\') { p++; if (p >= e) bad(); char c = *p++;
        switch (c) { case 'n': r += '\n'; break; case 't': r += '\t'; break; case 'r': r += '\r'; break;
          case 'b': r += '\b'; break; case 'f': r += '\f'; break;
          case 'u': { if (e - p < 4) bad(); unsigned v = strtoul(std::string(p, 4).c_str(), nullptr, 16); p += 4;
            if (v < 0x80) r += (char)v; else if (v < 0x800) { r += (char)(0xC0 | (v >> 6)); r += (char)(0x80 | (v & 63)); }
            else { r += (char)(0xE0 | (v >> 12)); r += (char)(0x80 | ((v >> 6) & 63)); r += (char)(0x80 | (v & 63)); } break; }
          default: r += c; }
      } else r += *p++;
    }
    if (p >= e) bad(); p++; return r;
  }
  J val() {
    ws(); if (p >= e) bad(); J v;
    if (*p == '{') { v.t = J::OBJ; p++; ws(); if (*p == '}') { p++; return v; }
      for (;;) { ws(); std::string k = str(); ws(); if (*p != ':') bad(); p++; v.o.emplace_back(k, val()); ws();
        if (*p == ',') { p++; continue; } if (*p == '}') { p++; break; } bad(); } }
    else if (*p == '[') { v.t = J::ARR; p++; ws(); if (*p == ']') { p++; return v; }
      for (;;) { v.a.push_back(val()); ws(); if (*p == ',') { p++; continue; } if (*p == ']') { p++; break; } bad(); } }
    else if (*p == '"') { v.t = J::STR; v.s = str(); }
    else if (!strncmp(p, "true", 4)) { v.t = J::BOOL; v.b = true; p += 4; }
    else if (!strncmp(p, "false", 5)) { v.t = J::BOOL; p += 5; }
    else if (!strncmp(p, "null", 4)) { p += 4; }
    else { v.t = J::NUM; const char* q = p; bool fl = false;
      if (*q == '-') q++; while (q < e && ((*q >= '0' && *q <= '9') || *q == '.' || *q == 'e' || *q == 'E' || *q == '+' || *q == '-')) {
        if (*q == '.' || *q == 'e' || *q == 'E') fl = true; q++; }
      std::string t(p, q); if (t.empty()) bad(); v.n = strtod(t.c_str(), nullptr);
      if (!fl) { v.i = strtoll(t.c_str(), nullptr, 10); v.isint = true; } else v.i = (int64_t)v.n; p = q; }
    return v;
  }
};
static J parse_json(const char* s, size_t n) { JP jp{s, s + n}; J v = jp.val(); return v; }

// ------------------------------------------------------------------------------------------------ safetensors
struct TInfo { std::string dtype; std::vector<int64_t> shape; const uint8_t* data; size_t nbytes; };
struct Ckpt {
  std::map<std::string, TInfo> t;
  void open_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY); if (fd < 0) die("cannot open " + path);
    struct stat st; fstat(fd, &st);
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); if (m == MAP_FAILED) die("mmap " + path);
    ::close(fd);
    const uint8_t* base = (const uint8_t*)m; uint64_t hl; memcpy(&hl, base, 8);
    J h = parse_json((const char*)base + 8, hl);
    for (auto& kv : h.o) {
      if (kv.first == "__metadata__") continue;
      TInfo ti; ti.dtype = kv.second.get("dtype")->s;
      for (auto& d : kv.second.get("shape")->a) ti.shape.push_back(d.i);
      auto& off = kv.second.get("data_offsets")->a;
      ti.data = base + 8 + hl + off[0].i; ti.nbytes = off[1].i - off[0].i;
      t[kv.first] = ti;
    }
  }
  const TInfo& get(const std::string& k) const { auto it = t.find(k); if (it == t.end()) die("missing tensor " + k); return it->second; }
  const uint16_t* bf(const std::string& k) const { auto& x = get(k); if (x.dtype != "BF16") die(k + " not bf16"); return (const uint16_t*)x.data; }
  std::vector<float> f32(const std::string& k) const {
    auto& x = get(k); size_t n = x.nbytes / 2; std::vector<float> r(n); const uint16_t* s = (const uint16_t*)x.data;
    for (size_t i = 0; i < n; i++) r[i] = bf2f(s[i]); return r;
  }
};

// MXFP4 row decoder: row r of an expert tensor (blocks [R][G][16] u8, scales [R][G] u8) -> fp8 e5m2 bytes [G*32].
// Exact: |fp4| = 2^a * {1, 1.5} with a in [-1, 2], and this checkpoint's scales are 2^-12..2^9, so every value is
// 2^b * {1, 1.5} with b in [-13, 11], inside e5m2's normal range [2^-14, 57344] with one of its two mantissa bits.
static const float FP4[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
static uint8_t f2e5m2(float v) {
  uint32_t u; memcpy(&u, &v, 4);
  uint8_t s = (u >> 24) & 0x80;
  if ((u & 0x7FFFFFFF) == 0) return s;
  int e = (int)((u >> 23) & 0xFF) - 127 + 15; uint32_t m = u & 0x7FFFFF;
  if (e < 1 || e > 30 || (m & 0x1FFFFF)) die("MXFP4 value not exactly representable in fp8 e5m2");
  return s | (uint8_t)(e << 2) | (uint8_t)(m >> 21);
}
static void mx_row(const uint8_t* blocks, const uint8_t* scales, int G, uint8_t* out) {
  for (int g = 0; g < G; g++) {
    float sc = ldexpf(1.0f, (int)scales[g] - 127);
    uint8_t lut[16];
    for (int i = 0; i < 16; i++) lut[i] = f2e5m2(FP4[i] * sc);
    const uint8_t* b = blocks + g * 16;
    for (int j = 0; j < 16; j++) { out[g * 32 + 2 * j] = lut[b[j] & 15]; out[g * 32 + 2 * j + 1] = lut[b[j] >> 4]; }
  }
}

// ------------------------------------------------------------------------------------------------ device state
struct Model {
  nrt_model_t* m = nullptr; std::vector<std::string> ins, outs;
};
static Model load_model(const char* b, const char* e, int vnc, bool coll) {
  Model M; size_t sz = e - b;
  if (coll) CK(nrt_load_collectives(b, sz, vnc, -1, vnc, W, &M.m));
  else CK(nrt_load(b, sz, vnc, -1, &M.m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(M.m, &info));
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& t = info->tensor_array[i];
    (t.usage == NRT_TENSOR_USAGE_INPUT ? M.ins : M.outs).push_back(t.name);
  }
  CK(nrt_free_model_tensor_info(info));
  return M;
}

struct LayerT { std::map<std::string, nrt_tensor_t*> t; };
// Layer tensors: one device tensor per name stacked over all NL layers (the fused decode NEFF indexes the layer at a
// compile-time offset); each layer's part is a slice (LayerT) for the per-layer prefill NEFFs.
static const int T_ = MAXPOS, XS = 12;  // XS: layers in the first expert-weight stack (kernels/model.py)
static const std::vector<std::pair<std::string, size_t>> LAYER_BYTES = {
  {"ln1", (size_t)P * HT * 4}, {"ln2", (size_t)P * HT * 4}, {"bo", (size_t)P * HT * 4},
  {"wqkv", (size_t)2 * P * QKV_F * 2}, {"bqkv", (size_t)2 * D * QKV_T * 4}, {"wo", (size_t)2 * D * WO_F * 2}, {"wo2", (size_t)2 * 128 * WO_F / 2 * 2},
  {"sinks", (size_t)2 * NQ * 4}, {"wr", (size_t)P * HT * E * 2}, {"br", (size_t)E * 4},
  {"wgu", (size_t)E * 2 * P * GU_F}, {"bgu", (size_t)E * 2 * P * GU_T * 4}, {"bgut", (size_t)E * 2 * GU_T * P * 4}, {"wd", (size_t)E * 2 * P * DN_F},
  {"bd", (size_t)E * H * 4}, {"kc", (size_t)2 * D * T_ * 2}, {"vc", (size_t)2 * T_ * D * 2},
  // decode copies padded to 128 partitions (rows 120..127 zero): 128-partition DMAs stream at ~2x the rate
  {"wqkv8", (size_t)2 * 128 * QKV_F * 2}, {"wgu8", (size_t)E * 2 * 128 * GU_F}, {"wd8", (size_t)E * 2 * 128 * DN_F}};
static bool is_expert_stack(const std::string& n) { return n == "wgu" || n == "wd" || n == "wgu8" || n == "wd8"; }
struct Rank {
  int vnc = 0;
  Model head; std::vector<Model> step;
  Model pembed, pgather, player_s, player_f512, pdense; std::vector<Model> pattn_s, pattn_f, pmoe;
  nrt_tensor_t *cpos[NCH], *dummy;
  nrt_tensor_set_t *pembed_in[NCH], *pembed_out, *pgather_in[2], *pgather_out, *pattn_in[NL], *pattn_out, *pmoe_in[NL], *pmoe_out, *player_in[NL], *player_out, *pdense_in, *pdense_out;
  std::map<std::string, nrt_tensor_t*> g;  // global tensors
  nrt_tensor_t *pos[2], *h[2], *tokout;
  LayerT L[NL];
  std::map<std::string, nrt_tensor_t*> stack;  // stacked layer tensors (LAYER_BYTES)
  nrt_tensor_set_t *step_in[2], *step_out[2], *head_in[2], *head_out[2];
  std::mutex mu;
};
static Rank R[W];
static bool g_sliding[NL];

static nrt_tensor_t* dev_alloc(int vnc, size_t bytes, const std::string& name) {
  nrt_tensor_t* t; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, vnc, bytes, name.c_str(), &t)); return t;
}
static void upload(Rank& r, nrt_tensor_t* t, const void* data, size_t bytes) {
  const size_t CH = 256u << 20;  // chunked writes
  for (size_t o = 0; o < bytes; o += CH) CK(nrt_tensor_write(t, (const char*)data + o, o, std::min(CH, bytes - o)));
  static bool verify = getenv("ENGINE_VERIFY") != nullptr;  // debug: read back every upload
  if (verify) {
    std::vector<char> back(std::min(CH, bytes));
    for (size_t o = 0; o < bytes; o += CH) {
      size_t n = std::min(CH, bytes - o); CK(nrt_tensor_read(t, back.data(), o, n));
      if (memcmp(back.data(), (const char*)data + o, n)) { fprintf(stderr, "[engine] VERIFY MISMATCH rank %d at %zu\n", r.vnc, o); }
    }
  }
}
static nrt_tensor_t* layer_slice(Rank& r, int l, const std::string& name, size_t bytes) {
  size_t lb = 0; for (auto& kv : LAYER_BYTES) if (kv.first == name) lb = kv.second;
  if (lb != bytes) die("layer tensor " + name + " has " + std::to_string(bytes) + " bytes, expected " + std::to_string(lb));
  std::string sn = name; int ls = l;
  if (is_expert_stack(name) && l >= XS) { sn = name + "2"; ls = l - XS; }  // second expert stack
  auto it = r.stack.find(sn); if (it == r.stack.end()) die("no stacked tensor " + sn);
  nrt_tensor_t* t; CK(nrt_tensor_allocate_slice(it->second, (size_t)ls * lb, lb, ("L" + std::to_string(l) + "_" + name).c_str(), &t));
  return t;
}
template <class T> static nrt_tensor_t* lput(Rank& r, int l, const std::string& name, const std::vector<T>& v) {
  nrt_tensor_t* t = layer_slice(r, l, name, v.size() * sizeof(T)); upload(r, t, v.data(), v.size() * sizeof(T)); return t;
}
template <class T> static nrt_tensor_t* put(Rank& r, const std::string& name, const std::vector<T>& v) {
  nrt_tensor_t* t = dev_alloc(r.vnc, v.size() * sizeof(T), name); upload(r, t, v.data(), v.size() * sizeof(T)); return t;
}
static nrt_tensor_t* zeros(Rank& r, const std::string& name, size_t bytes) {
  std::vector<uint8_t> z(bytes, 0); nrt_tensor_t* t = dev_alloc(r.vnc, bytes, name); upload(r, t, z.data(), bytes); return t;
}

static nrt_tensor_set_t* make_set(const std::vector<std::string>& names, const std::map<std::string, nrt_tensor_t*>& m,
                                  const char* what) {
  nrt_tensor_set_t* s; CK(nrt_allocate_tensor_set(&s));
  for (auto& n : names) {
    auto it = m.find(n); if (it == m.end()) die(std::string("no tensor for ") + what + " input/output " + n);
    CK(nrt_add_tensor_to_tensor_set(s, n.c_str(), it->second));
  }
  return s;
}

static std::vector<float> col(const std::vector<float>& v) {  // [H] -> [P][HT]
  std::vector<float> r(H); for (int kt = 0; kt < HT; kt++) for (int p = 0; p < P; p++) r[p * HT + kt] = v[kt * P + p]; return r;
}

// RoPE table [MAXPOS][128] fp32: cos x2 | sin x2, YaRN (factor 32, beta 32/1, orig 4096, theta 150000), as HF
static std::vector<float> rope_table() {
  const int dim = 64; const double base = 150000.0, factor = 32.0, orig = 4096.0, bfast = 32.0, bslow = 1.0;
  double attn = 0.1 * std::log(factor) + 1.0;
  auto fcd = [&](double rot) { return (dim * std::log(orig / (rot * 2 * M_PI))) / (2 * std::log(base)); };
  double low = std::max(fcd(bfast), 0.0), high = std::min(fcd(bslow), (double)dim - 1);
  float inv[32];
  for (int i = 0; i < 32; i++) {
    float ex = (float)(2 * i) / (float)dim;
    float pf = (float)std::pow((double)(float)base, (double)ex);
    float extra = 1.0f / pf, inter = 1.0f / ((float)factor * pf);
    float ramp = ((float)i - (float)low) / ((float)high - (float)low);
    ramp = std::min(std::max(ramp, 0.0f), 1.0f);
    float ef = 1.0f - ramp;
    inv[i] = inter * (1.0f - ef) + extra * ef;
  }
  std::vector<float> t((size_t)MAXPOS * 128);
  for (int p = 0; p < MAXPOS; p++)
    for (int i = 0; i < 32; i++) {
      float fr = inv[i] * (float)p;
      float c = (float)std::cos((double)fr) * (float)attn, s = (float)std::sin((double)fr) * (float)attn;
      c = bf2f(f2bf(c)); s = bf2f(f2bf(s));
      float* row = &t[(size_t)p * 128];
      row[i] = row[32 + i] = c; row[64 + i] = row[96 + i] = s;
    }
  return t;
}

static void parallel_for(int n, int nthreads, const std::function<void(int)>& f) {
  std::atomic<int> next(0); std::vector<std::thread> th; std::mutex em; std::string err;
  for (int k = 0; k < nthreads; k++) th.emplace_back([&] {
    for (;;) { int i = next++; if (i >= n) break;
      try { f(i); } catch (std::exception& ex) { std::lock_guard<std::mutex> lk(em); err = ex.what(); } } });
  for (auto& t : th) t.join();
  if (!err.empty()) throw std::runtime_error(err);
}

static void load_layer(const Ckpt& ck, int l) {
  std::string pre = "model.layers." + std::to_string(l) + ".";
  const uint16_t* qw = ck.bf(pre + "self_attn.q_proj.weight"); const uint16_t* kw = ck.bf(pre + "self_attn.k_proj.weight");
  const uint16_t* vw = ck.bf(pre + "self_attn.v_proj.weight"); const uint16_t* ow = ck.bf(pre + "self_attn.o_proj.weight");
  auto qb = ck.f32(pre + "self_attn.q_proj.bias"), kb = ck.f32(pre + "self_attn.k_proj.bias"), vb = ck.f32(pre + "self_attn.v_proj.bias");
  auto sk = ck.f32(pre + "self_attn.sinks");
  auto ln1 = col(ck.f32(pre + "input_layernorm.weight")), ln2 = col(ck.f32(pre + "post_attention_layernorm.weight"));
  auto bo = col(ck.f32(pre + "self_attn.o_proj.bias"));
  const uint16_t* wrw = ck.bf(pre + "mlp.router.weight"); auto br = ck.f32(pre + "mlp.router.bias");
  auto gub = ck.f32(pre + "mlp.experts.gate_up_proj_bias"); auto bd = ck.f32(pre + "mlp.experts.down_proj_bias");
  auto& gblk = ck.get(pre + "mlp.experts.gate_up_proj_blocks"); auto& gsc = ck.get(pre + "mlp.experts.gate_up_proj_scales");
  auto& dblk = ck.get(pre + "mlp.experts.down_proj_blocks"); auto& dsc = ck.get(pre + "mlp.experts.down_proj_scales");
  if (gblk.shape != std::vector<int64_t>{E, 2 * H, H / 32, 16} || dblk.shape != std::vector<int64_t>{E, H, H / 32, 16})
    die("unexpected expert block shapes");
  // wr [P][HT][E]
  std::vector<uint16_t> wr((size_t)P * HT * E);
  for (int p = 0; p < P; p++) for (int kt = 0; kt < HT; kt++) for (int e = 0; e < E; e++)
    wr[(p * HT + kt) * E + e] = wrw[(size_t)e * H + kt * P + p];
  // experts for all ranks: wgu [E][2][P][HT][6][P] (kt-major), wd [E][2][P][3][HT][P]
  std::vector<std::vector<uint8_t>> wgu(W), wd(W);  // fp8 e5m2 (exact, mx_row)
  std::vector<std::vector<float>> bgu(W), bgut(W);
  for (int r = 0; r < W; r++) { wgu[r].resize((size_t)E * 2 * P * GU_F); wd[r].resize((size_t)E * 2 * P * DN_F); bgu[r].resize((size_t)E * 2 * P * GU_T); bgut[r].resize((size_t)E * 2 * GU_T * P); }
  std::vector<uint8_t> row(H);
  for (int e = 0; e < E; e++) {
    for (int rr = 0; rr < 2 * H; rr++) {  // gate_up rows: rr = 2j (+1 for up)
      mx_row(gblk.data + ((size_t)e * 2 * H + rr) * 90 * 16, gsc.data + ((size_t)e * 2 * H + rr) * 90, 90, row.data());
      int j = rr / 2, up = rr & 1, g = j / 360, jj = j % 360, t = (jj / P) + 3 * up, n = jj % P;
      int rank = g / 2, c = g % 2;
      uint8_t* dst = &wgu[rank][(((size_t)e * 2 + c) * P) * GU_F];
      for (int k = 0; k < H; k++) { int p = k % P, kt = k / P; dst[(size_t)p * GU_F + (kt * GU_T + t) * P + n] = row[k]; }
      bgu[rank][(((size_t)e * 2 + c) * P + n) * GU_T + t] = gub[(size_t)e * 2 * H + rr];
      bgut[rank][(((size_t)e * 2 + c) * GU_T + t) * P + n] = gub[(size_t)e * 2 * H + rr];
    }
    for (int hr = 0; hr < H; hr++) {  // down rows: output feature hr, along the intermediate dim
      mx_row(dblk.data + ((size_t)e * H + hr) * 90 * 16, dsc.data + ((size_t)e * H + hr) * 90, 90, row.data());
      int nt = hr / P, n = hr % P;
      for (int i = 0; i < H; i++) {
        int g = i / 360, ii = i % 360, it = ii / P, p = ii % P, rank = g / 2, c = g % 2;
        wd[rank][(((size_t)e * 2 + c) * P + p) * DN_F + (it * HT + nt) * P + n] = row[i];
      }
    }
  }
  for (int r = 0; r < W; r++) {
    Rank& rk = R[r]; LayerT& L = rk.L[l];
    std::vector<uint16_t> wqkv((size_t)2 * P * QKV_F), wo((size_t)2 * D * WO_F), wo2((size_t)2 * 128 * WO_F / 2);
    std::vector<float> bqkv(2 * D * QKV_T), sinks(2 * NQ);
    for (int c = 0; c < 2; c++) {
      int g = 2 * r + c;
      for (int t = 0; t < QKV_T; t++) for (int n = 0; n < D; n++) {
        const uint16_t* src = t < 8 ? qw + (size_t)(512 * g + t * 64 + n) * H : t == 8 ? kw + (size_t)(64 * g + n) * H : vw + (size_t)(64 * g + n) * H;
        for (int k = 0; k < H; k++) wqkv[((size_t)c * P + k % P) * QKV_F + ((k / P) * QKV_T + t) * D + n] = src[k];  // [p][kt][t][n]
        bqkv[(c * D + n) * QKV_T + t] = t < 8 ? qb[512 * g + t * 64 + n] : t == 8 ? kb[64 * g + n] : vb[64 * g + n];
      }
      for (int out = 0; out < H; out++) for (int hh = 0; hh < NQ; hh++) for (int d = 0; d < D; d++)
        wo[((size_t)c * D + d) * WO_F + (hh * HT + out / P) * P + out % P] = ow[(size_t)out * 4096 + 512 * g + hh * 64 + d];
      // decode layout: head pairs on 128 partitions, row (hh % 2) * 64 + d, column (hh / 2) * H + out
      for (int out = 0; out < H; out++) for (int hh = 0; hh < NQ; hh++) for (int d = 0; d < D; d++)
        wo2[((size_t)c * 128 + (hh % 2) * 64 + d) * (WO_F / 2) + (hh / 2) * H + out] = ow[(size_t)out * 4096 + 512 * g + hh * 64 + d];
      for (int i = 0; i < NQ; i++) sinks[c * NQ + i] = sk[8 * g + i];
    }
    std::string s = "L" + std::to_string(l) + "_";
    if (const char* dd = getenv("ENGINE_DUMP_DIR")) if (l == atoi(getenv("ENGINE_DUMP_LAYER") ? getenv("ENGINE_DUMP_LAYER") : "0")) {
      auto dump = [&](const std::string& n, const void* p, size_t b) {
        std::string f = std::string(dd) + "/r" + std::to_string(r) + "_" + n + ".bin"; FILE* fp = fopen(f.c_str(), "wb"); fwrite(p, 1, b, fp); fclose(fp); };
      dump("wqkv", wqkv.data(), wqkv.size() * 2); dump("bqkv", bqkv.data(), bqkv.size() * 4); dump("wo", wo.data(), wo.size() * 2);
      dump("sinks", sinks.data(), sinks.size() * 4); dump("wr", wr.data(), wr.size() * 2); dump("br", br.data(), br.size() * 4);
      dump("wgu", wgu[r].data(), wgu[r].size()); dump("bgu", bgu[r].data(), bgu[r].size() * 4); dump("wd", wd[r].data(), wd[r].size());
      dump("bd", bd.data(), bd.size() * 4); dump("ln1", ln1.data(), ln1.size() * 4); dump("ln2", ln2.data(), ln2.size() * 4); dump("bo", bo.data(), bo.size() * 4);
    }
    L.t["ln1"] = lput(rk, l, "ln1", ln1); L.t["ln2"] = lput(rk, l, "ln2", ln2); L.t["bo"] = lput(rk, l, "bo", bo);
    L.t["wqkv"] = lput(rk, l, "wqkv", wqkv); L.t["bqkv"] = lput(rk, l, "bqkv", bqkv); L.t["wo"] = lput(rk, l, "wo", wo); L.t["wo2"] = lput(rk, l, "wo2", wo2);
    L.t["sinks"] = lput(rk, l, "sinks", sinks); L.t["wr"] = lput(rk, l, "wr", wr); L.t["br"] = lput(rk, l, "br", br);
    {  // decode copies padded to 128 partitions
      auto pad = [](const std::vector<uint8_t>& v, size_t blocks, size_t F) {
        std::vector<uint8_t> o(blocks * 128 * F, 0);
        for (size_t b = 0; b < blocks; b++) memcpy(&o[b * 128 * F], &v[b * P * F], (size_t)P * F);
        return o;
      };
      // wgu8 per kt in the column-tiled order [it][up][n] (kernels/model.py CT_GU) instead of [up][it][n]
      std::vector<uint8_t> g8((size_t)E * 2 * 128 * GU_F, 0);
      for (size_t b = 0; b < (size_t)E * 2; b++) for (int p = 0; p < P; p++) {
        const uint8_t* src = &wgu[r][(b * P + p) * GU_F]; uint8_t* dst = &g8[(b * 128 + p) * GU_F];
        for (int kt = 0; kt < HT; kt++) for (int it = 0; it < IT; it++) for (int up = 0; up < 2; up++)
          memcpy(dst + (size_t)(kt * GU_T + it * 2 + up) * P, src + (size_t)(kt * GU_T + it + 3 * up) * P, P);
      }
      L.t["wgu8"] = lput(rk, l, "wgu8", g8); L.t["wd8"] = lput(rk, l, "wd8", pad(wd[r], (size_t)E * 2, DN_F));
      std::vector<uint16_t> q8((size_t)2 * 128 * QKV_F, 0);
      for (int c = 0; c < 2; c++) memcpy(&q8[(size_t)c * 128 * QKV_F], &wqkv[(size_t)c * P * QKV_F], (size_t)P * QKV_F * 2);
      lput(rk, l, "wqkv8", q8);
    }
    L.t["wgu"] = lput(rk, l, "wgu", wgu[r]); L.t["bgu"] = lput(rk, l, "bgu", bgu[r]); L.t["bgut"] = lput(rk, l, "bgut", bgut[r]); L.t["wd"] = lput(rk, l, "wd", wd[r]);
    L.t["bd"] = lput(rk, l, "bd", bd);
    int T = MAXPOS;
    L.t["kc"] = lput(rk, l, "kc", std::vector<uint16_t>((size_t)2 * D * T, 0));
    L.t["vc"] = lput(rk, l, "vc", std::vector<uint16_t>((size_t)2 * T * D, 0));
    std::vector<uint8_t>().swap(wgu[r]); std::vector<uint8_t>().swap(wd[r]);
  }
}

static void load_head(const Ckpt& ck, int r) {
  Rank& rk = R[r];
  const uint16_t* W_ = ck.bf("lm_head.weight");
  std::vector<uint16_t> wlm((size_t)2 * NG * 128 * LM_F, 0);  // [2][NG][128][LM_G][HT2][128], rows k >= H zero
  std::vector<float> lb((size_t)2 * 128 * NVT), ls((size_t)2 * 128 * NVT);
  for (int c = 0; c < 2; c++) {
    int g = 2 * r + c; int64_t base = (int64_t)g * NVT * 128;
    for (int vt = 0; vt < NVT; vt++) for (int n = 0; n < 128; n++) {
      int64_t id = base + (int64_t)vt * 128 + n;
      lb[((size_t)c * 128 + n) * NVT + vt] = id < VOCAB ? 0.0f : -3e38f;
      ls[((size_t)c * 128 + n) * NVT + vt] = BIG - (float)id;
      if (id >= VOCAB) continue;
      int grp = vt / LM_G, j = vt % LM_G;
      const uint16_t* src = W_ + (size_t)id * H;
      uint16_t* dst = &wlm[((size_t)c * NG + grp) * 128 * LM_F];
      for (int k = 0; k < H; k++) dst[(size_t)(k % 128) * LM_F + (j * HT2 + k / 128) * 128 + n] = src[k];
    }
  }
  if (const char* dd = getenv("ENGINE_DUMP_DIR")) {
    auto dump = [&](const std::string& n, const void* p, size_t b) {
      std::string f = std::string(dd) + "/r" + std::to_string(r) + "_" + n + ".bin"; FILE* fp = fopen(f.c_str(), "wb"); fwrite(p, 1, b, fp); fclose(fp); };
    dump("wlm", wlm.data(), wlm.size() * 2); dump("lmbias", lb.data(), lb.size() * 4); dump("lmscore", ls.data(), ls.size() * 4);
    auto lc = col(ck.f32("model.norm.weight")); dump("lnf", lc.data(), lc.size() * 4);
  }
  rk.g["wlm"] = put(rk, "wlm", wlm); rk.g["lmbias"] = put(rk, "lmbias", lb); rk.g["lmscore"] = put(rk, "lmscore", ls);
  rk.g["lnf"] = put(rk, "lnf", col(ck.f32("model.norm.weight")));
  auto& emb = ck.get("model.embed_tokens.weight");
  nrt_tensor_t* te = dev_alloc(rk.vnc, emb.nbytes, "emb"); upload(rk, te, emb.data, emb.nbytes); rk.g["emb"] = te;
}

static int bucket_index(int pos) {
  for (size_t i = 0; i < FULL_BUCKETS.size(); i++) if (pos < FULL_BUCKETS[i]) return (int)i;
  return -1;
}

static void setup(const std::string& mdir) {
  auto t0 = clk::now();
  // config sanity
  {
    int fd = ::open((mdir + "/config.json").c_str(), O_RDONLY); if (fd < 0) die("no config.json");
    std::string s; char buf[65536]; ssize_t n; while ((n = read(fd, buf, sizeof buf)) > 0) s.append(buf, n); ::close(fd);
    J c = parse_json(s.data(), s.size());
    if (c.get("hidden_size")->i != H || c.get("num_hidden_layers")->i != NL || c.get("vocab_size")->i != VOCAB)
      die("config does not match the compiled model");
    auto& lt = c.get("layer_types")->a;
    for (int l = 0; l < NL; l++) g_sliding[l] = lt[l].s == "sliding_attention";
    for (int l = 0; l < NL; l++) if (g_sliding[l] != (l % 2 == 0)) die("the decode NEFF assumes alternating sliding / full layers");
  }
  Ckpt ck;
  {
    int fd = ::open((mdir + "/model.safetensors.index.json").c_str(), O_RDONLY); if (fd < 0) die("no index");
    std::string s; char buf[65536]; ssize_t n; while ((n = read(fd, buf, sizeof buf)) > 0) s.append(buf, n); ::close(fd);
    J idx = parse_json(s.data(), s.size()); std::map<std::string, int> files;
    for (auto& kv : idx.get("weight_map")->o) files[kv.second.s] = 1;
    for (auto& f : files) ck.open_file(mdir + "/" + f.first);
  }
  // ranks
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  uint32_t nvis; CK(nrt_get_visible_vnc_count(&nvis));
  if (nvis < (uint32_t)W) die("need 4 visible logical NeuronCores, have " + std::to_string(nvis));
  for (int r = 0; r < W; r++) R[r].vnc = r;
  parallel_for(W, W, [&](int r) {
    Rank& rk = R[r];
    rk.head = load_model(neff_head_start, neff_head_end, r, true);
    rk.step.push_back(load_model(neff_step384_start, neff_step384_end, r, true));
    rk.step.push_back(load_model(neff_step512_start, neff_step512_end, r, true));
    rk.step.push_back(load_model(neff_step1152_start, neff_step1152_end, r, true));
    rk.step.push_back(load_model(neff_step2048_start, neff_step2048_end, r, true));
    rk.step.push_back(load_model(neff_step4224_start, neff_step4224_end, r, true));
    rk.step.push_back(load_model(neff_step4608_start, neff_step4608_end, r, true));
    rk.pembed = load_model(neff_pembed_start, neff_pembed_end, r, false);
    rk.pgather = load_model(neff_pgather_start, neff_pgather_end, r, false);
#define LM(v, n) rk.v.push_back(load_model(neff_##n##_start, neff_##n##_end, r, true))
    LM(pattn_s, pattfa_s2); LM(pattn_s, pattfa_s4); LM(pattn_s, pattfa_s8); LM(pattn_s, pattfa_s12);
    LM(pattn_s, pattfa_s16); LM(pattn_s, pattfa_s18);
    LM(pattn_f, pattfa_f2); LM(pattn_f, pattfa_f4); LM(pattn_f, pattfa_f8); LM(pattn_f, pattfa_f12);
    LM(pattn_f, pattfa_f16); LM(pattn_f, pattfa_f18);
    LM(pmoe, pmoe8); LM(pmoe, pmoe16); LM(pmoe, pmoe24); LM(pmoe, pmoe32); LM(pmoe, pmoe36);
#undef LM
    rk.player_s = load_model(neff_player_s_start, neff_player_s_end, r, true);
    rk.player_f512 = load_model(neff_player_f512_start, neff_player_f512_end, r, true);
    rk.pdense = load_model(neff_pdense_start, neff_pdense_end, r, true);
  });
  fprintf(stderr, "[engine] NEFFs loaded %.0f ms\n", ms_since(t0));
  auto rope = rope_table();
  std::vector<float> ropeT((size_t)128 * MAXPOS);
  for (int p = 0; p < MAXPOS; p++) for (int i = 0; i < 128; i++) ropeT[(size_t)i * MAXPOS + p] = rope[(size_t)p * 128 + i];
  std::vector<uint16_t> rm(D * D, 0);
  for (int d = 0; d < 32; d++) { rm[(d + 32) * D + d] = f2bf(-1.f); rm[d * D + d + 32] = f2bf(1.f); }
  parallel_for(W, W, [&](int r) {
    Rank& rk = R[r];
    rk.g["toks"] = zeros(rk, "toks", (size_t)TOKLEN * 4);
    rk.pos[0] = zeros(rk, "posA", 4); rk.pos[1] = zeros(rk, "posB", 4);
    rk.h[0] = zeros(rk, "hA", H * 2); rk.h[1] = zeros(rk, "hB", H * 2); rk.tokout = zeros(rk, "tokout", 4);
    rk.g["cs"] = put(rk, "cs", rope); rk.g["rmat"] = put(rk, "rmat", rm);
    rk.g["csT"] = put(rk, "csT", ropeT);
    rk.g["hT"] = zeros(rk, "hT", (size_t)H * MAXPOS * 2);
    // exchange scratch between the two physical cores: [2, 2, P, HT * SC] fp32 (prefill_attn), [2, MAXPOS, H] fp32
    // (prefill_moe)
    rk.g["xch"] = zeros(rk, "xch", std::max((size_t)2 * 2 * P * HT * SC, (size_t)2 * MAXPOS * H) * 4);
    rk.g["xtm"] = zeros(rk, "xtm", (size_t)MAXPOS * H * 2);
    rk.g["rti"] = zeros(rk, "rti", (size_t)MAXPOS * 8 * 4);
    rk.g["rdn"] = zeros(rk, "rdn", (size_t)MAXPOS * E * 4);
    rk.g["rec"] = zeros(rk, "rec", (size_t)E * (MAXPOS + 512) * 4 * 4);
    rk.g["blist"] = zeros(rk, "blist", (size_t)(4 * 36 + 2) * 4 * 4);
    rk.g["ybuf"] = zeros(rk, "ybuf", (size_t)2 * 4 * MAXPOS * H * 2);  // bf16
    rk.g["plen"] = zeros(rk, "plen", 4);
    rk.g["qbuf"] = zeros(rk, "qbuf", (size_t)2 * NQ * D * MAXPOS * 2);  // flash-attention prefill scratch
    rk.g["obuf"] = zeros(rk, "obuf", (size_t)2 * NQ * D * MAXPOS * 2);
    rk.g["kbuf"] = zeros(rk, "kbuf", (size_t)2 * D * MAXPOS * 2);
    rk.g["vbuf"] = zeros(rk, "vbuf", (size_t)2 * MAXPOS * D * 2);
    rk.dummy = zeros(rk, "dummy", 4);
    for (int c = 0; c < NCH; c++) { int32_t v = c * SC; rk.cpos[c] = put(rk, "cpos" + std::to_string(c), std::vector<int32_t>{v}); }
    load_head(ck, r);
    for (auto& kv : LAYER_BYTES) {
      bool x = is_expert_stack(kv.first);  // expert weights: two stacks of XS and NL - XS layers
      rk.stack[kv.first] = dev_alloc(rk.vnc, kv.second * (x ? XS : NL), "S_" + kv.first);
      if (x) rk.stack[kv.first + "2"] = dev_alloc(rk.vnc, kv.second * (NL - XS), "S_" + kv.first + "2");
    }
  });
  fprintf(stderr, "[engine] head/embedding uploaded %.0f ms\n", ms_since(t0));
  unsigned hw = std::thread::hardware_concurrency();
  parallel_for(NL, std::min<int>(NL, std::max(4u, hw / 8)), [&](int l) { load_layer(ck, l); });
  fprintf(stderr, "[engine] layers uploaded %.0f ms\n", ms_since(t0));
  // tensor sets
  for (int r = 0; r < W; r++) {
    Rank& rk = R[r];
    for (int q = 0; q < 2; q++) {
      auto m = rk.g; m["pos"] = rk.pos[q];
      m["h"] = rk.h[0];
      rk.head_in[q] = make_set(rk.head.ins, m, "head");
      std::map<std::string, nrt_tensor_t*> o{{rk.head.outs[0], rk.pos[1 - q]}, {rk.head.outs.size() > 1 ? rk.head.outs[1] : "-", rk.tokout}};
      rk.head_out[q] = make_set(rk.head.outs, o, "head");
      for (auto& kv : rk.stack) m[kv.first] = kv.second;
      m["kcw"] = rk.stack["kc"]; m["vcw"] = rk.stack["vc"];  // the decode step writes the caches through these aliases
      m["wqkv"] = rk.stack["wqkv8"]; m["wgu"] = rk.stack["wgu8"]; m["wd"] = rk.stack["wd8"];  // 128-partition copies
      m["wgu2"] = rk.stack["wgu82"]; m["wd2"] = rk.stack["wd82"];
      rk.step_in[q] = make_set(rk.step[0].ins, m, "step");
      std::map<std::string, nrt_tensor_t*> so{{rk.step[0].outs[0], rk.pos[1 - q]}, {rk.step[0].outs.at(1), rk.tokout}};
      rk.step_out[q] = make_set(rk.step[0].outs, so, "step");
    }
    for (int c = 0; c < NCH; c++) {
      auto m = rk.g; m["pos"] = rk.cpos[c];
      rk.pembed_in[c] = make_set(rk.pembed.ins, m, "pembed");
      for (int l = 0; l < NL; l++) {
        auto ml = rk.L[l].t; for (auto& kv : rk.g) ml[kv.first] = kv.second; ml["pos"] = rk.cpos[c];
        if (c == 0) { ml["sinkf"] = rk.L[l].t["sinks"]; rk.pattn_in[l] = make_set(rk.pattn_s[0].ins, ml, "pattn"); }
        if (c == 0) rk.pmoe_in[l] = make_set(rk.pmoe[0].ins, ml, "pmoe");
        if (c == 0) rk.player_in[l] = make_set(rk.player_s.ins, ml, "player");
      }
    }
    rk.pembed_out = make_set(rk.pembed.outs, {{rk.pembed.outs[0], rk.dummy}}, "pembed");
    rk.pattn_out = make_set(rk.pattn_s[0].outs, {{rk.pattn_s[0].outs[0], rk.dummy}}, "pattn");
    rk.player_out = make_set(rk.player_s.outs, {{rk.player_s.outs[0], rk.dummy}}, "player");
    {  // all dense one-chunk layers in one NEFF: the stacked layer tensors (prefill layouts)
      auto m = rk.g; for (auto& kv : rk.stack) m[kv.first] = kv.second; m["pos"] = rk.cpos[0];
      rk.pdense_in = make_set(rk.pdense.ins, m, "pdense");
      rk.pdense_out = make_set(rk.pdense.outs, {{rk.pdense.outs[0], rk.dummy}}, "pdense");
    }
    rk.pmoe_out = make_set(rk.pmoe[0].outs, {{rk.pmoe[0].outs[0], rk.dummy}}, "pmoe");
    for (int q = 0; q < 2; q++) { auto m = rk.g; m["pos"] = rk.pos[q]; rk.pgather_in[q] = make_set(rk.pgather.ins, m, "pgather"); }
    rk.pgather_out = make_set(rk.pgather.outs, {{rk.pgather.outs[0], rk.h[0]}}, "pgather");
  }
  fprintf(stderr, "[engine] ready %.0f ms\n", ms_since(t0));
}

// ------------------------------------------------------------------------------------------------ execution
struct Step { int pos; bool logits; int prefill_len; };  // prefill_len > 0: chunked prefill of positions [0, len)
struct Exec { nrt_model_t* m; nrt_tensor_set_t* in; nrt_tensor_set_t* out; };
static int g_q = 0;      // which pos buffer holds the current position
static int g_cur = -1;   // position of the next token to consume (toks[g_cur] holds it); -1: no state

static void run_steps(const std::vector<Step>& steps) {
  std::vector<std::string> errs(W);
  std::vector<std::thread> th;
  for (int r = 0; r < W; r++) th.emplace_back([&, r] {
    try {
      Rank& rk = R[r]; std::vector<Exec> plan; plan.reserve(steps.size() * (NL + 2));
      int q = g_q;
      for (auto& s : steps) {
        if (s.prefill_len > 0) {  // chunks, then gather the last token's hidden state and the head (pos[q] = len - 1)
          // layer-major: all chunks' attention of layer l (chunk c reads the K/V of chunks < c), then the layer's
          // MoE over the whole prompt
          int nch = (s.prefill_len + SC - 1) / SC;
          for (int c = 0; c < nch; c++) plan.push_back({rk.pembed.m, rk.pembed_in[c], rk.pembed_out});
          static int pfl = getenv("ENGINE_PF_LAYERS") ? atoi(getenv("ENGINE_PF_LAYERS")) : NL;  // debug
          static int moe_min = getenv("ENGINE_MOE_MIN_TILES") ? atoi(getenv("ENGINE_MOE_MIN_TILES")) : 0;  // debug
          if (nch == 1 && pfl == NL && !getenv("ENGINE_NO_PDENSE")) {  // one chunk: all dense layers, one NEFF
            plan.push_back({rk.pdense.m, rk.pdense_in, rk.pdense_out});
          } else if (nch == 1) {  // one chunk: the dense layer per execution (debug: ENGINE_PF_LAYERS)
            for (int l = 0; l < pfl; l++)
              plan.push_back({g_sliding[l] ? rk.player_s.m : rk.player_f512.m, rk.player_in[l], rk.player_out});
          } else {
          int mi = 0; while (MOE_TILES[mi] * 128 < nch * SC || MOE_TILES[mi] < moe_min) mi++;
          int ai = 0; while (ATT_CHUNKS[ai] < nch) ai++;
          for (int l = 0; l < pfl; l++) {
            plan.push_back({g_sliding[l] ? rk.pattn_s[ai].m : rk.pattn_f[ai].m, rk.pattn_in[l], rk.pattn_out});
            plan.push_back({rk.pmoe[mi].m, rk.pmoe_in[l], rk.pmoe_out});
          }
          }
          plan.push_back({rk.pgather.m, rk.pgather_in[q], rk.pgather_out});
          plan.push_back({rk.head.m, rk.head_in[q], rk.head_out[q]});
          q = 1 - q;
          continue;
        }
        int bi = bucket_index(s.pos);
        plan.push_back({rk.step[bi].m, rk.step_in[q], rk.step_out[q]});
        q = 1 - q;
      }
      std::vector<NRT_STATUS> rets(plan.size(), NRT_SUCCESS); nrta_seq_t seq = 0;
      for (size_t i = 0; i < plan.size(); i++) {
        NRT_STATUS st;
        while ((st = nrta_execute_schedule(plan[i].m, plan[i].in, plan[i].out, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {}
        CK(st);
        static bool sync_each = getenv("ENGINE_SYNC_EACH") != nullptr;  // debug
        if (sync_each) { bool d = false; while (!d) CK(nrta_is_completed(seq, &d)); }
      }
      bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
      for (size_t i = 0; i < plan.size(); i++) if (rets[i] != NRT_SUCCESS)
        throw std::runtime_error("execution " + std::to_string(i) + " failed: " + nrt_get_status_as_str(rets[i]));
    } catch (std::exception& ex) { errs[r] = ex.what(); }
  });
  for (auto& t : th) t.join();
  for (auto& e : errs) if (!e.empty()) throw std::runtime_error(e);
  g_q = (g_q + (int)(steps.size() % 2)) % 2;
}

static std::vector<int> generate(const std::vector<int64_t>& prompt, int64_t max_tokens, bool reset) {
  if (max_tokens < 1) throw std::runtime_error("max_tokens must be at least 1");
  std::vector<Step> steps; int64_t first;
  if (reset) {
    if (prompt.empty()) throw std::runtime_error("a reset request needs a prompt");
    int64_t plen = prompt.size();
    if (plen + max_tokens > MAXPOS) throw std::runtime_error("prompt + max_tokens exceeds the compiled positions (" + std::to_string(MAXPOS) + ")");
    std::vector<int32_t> tk(((plen + SC - 1) / SC) * SC, 0);  // padded to whole prefill chunks with token 0
    for (int64_t i = 0; i < plen; i++) { if (prompt[i] < 0 || prompt[i] >= VOCAB) throw std::runtime_error("token id out of range"); tk[i] = (int32_t)prompt[i]; }
    g_cur = -1;
    int32_t last = (int32_t)(plen - 1);
    int32_t pl32 = (int32_t)plen;
    for (int r = 0; r < W; r++) {
      CK(nrt_tensor_write(R[r].g["toks"], tk.data(), 0, tk.size() * 4)); CK(nrt_tensor_write(R[r].pos[g_q], &last, 0, 4));
      CK(nrt_tensor_write(R[r].g["plen"], &pl32, 0, 4));
    }
    steps.push_back({0, true, (int)plen});
    for (int64_t k = 1; k < max_tokens; k++) steps.push_back({(int)(plen - 1 + k), true, 0});
    first = plen;
  } else {
    if (!prompt.empty() || g_cur < 0) throw std::runtime_error("a continuation has an empty prompt and follows a generate request");
    if (g_cur + max_tokens > MAXPOS) throw std::runtime_error("continuation exceeds the compiled positions (" + std::to_string(MAXPOS) + ")");
    for (int64_t k = 0; k < max_tokens; k++) steps.push_back({(int)(g_cur + k), true, 0});
    first = g_cur + 1;
  }
  g_cur = -1;
  static bool tim = getenv("ENGINE_TIMING") != nullptr;  // debug
  auto tr = clk::now();
  run_steps(steps);
  if (tim) fprintf(stderr, "[engine] run_steps %.3f ms\n", ms_since(tr));
  std::vector<int32_t> out(max_tokens);
  CK(nrt_tensor_read(R[0].g["toks"], out.data(), first * 4, max_tokens * 4));
  g_cur = (int)(first + max_tokens - 1);
  return std::vector<int>(out.begin(), out.end());
}

// The first execution of a NEFF in a process costs much more than later ones (measured: a 256-token prefill took
// 92.6 ms instead of 40 ms when its NEFFs had not run yet). Run every NEFF once at load: one prefill per prefill path
// (dense one-chunk layer, every attention bucket and MoE bucket) and decode steps in every attention bucket.
static void warmup() {
  if (getenv("ENGINE_NO_WARMUP")) return;  // debug
  auto t0 = clk::now();
  auto prompt = [](int n) { std::vector<int64_t> p(n); for (int i = 0; i < n; i++) p[i] = (int64_t)((i * 2654435761u) % 200000u); return p; };
  // prefill buckets (dense layer; attention chunks 2/4/8/12/16/18 with MoE tiles 4/8/16/24/32/36) and decode
  // attention buckets 384/512/1152/2048/4224/4608
  for (int n : {256, 400, 512, 600, 1024, 1500, 2048, 3072, 4096, 4400, MAXPOS - 8}) generate(prompt(n), 2, true);
  g_cur = -1;
  fprintf(stderr, "[engine] warmup %.0f ms\n", ms_since(t0));
}

// ------------------------------------------------------------------------------------------------ protocol
static std::string jesc(const std::string& s) {
  std::string r; for (char c : s) { if (c == '"' || c == '\\') { r += '\\'; r += c; } else if ((unsigned char)c < 0x20) r += ' '; else r += c; } return r;
}

int main(int argc, char** argv) {
  std::string mdir;
  for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--model") && i + 1 < argc) mdir = argv[++i];
  if (mdir.empty()) { fprintf(stderr, "usage: %s --model MODEL_DIR\n", argv[0]); return 2; }
  int fd = dup(1); g_out = fdopen(fd, "w"); dup2(2, 1);  // libraries that print go to stderr
  setenv("NEURON_RT_DISABLE_EXECUTION_BARRIER", "1", 0);
  {
    int c0 = 0; const char* vc = getenv("NEURON_RT_VISIBLE_CORES"); if (vc) c0 = atoi(vc);
    std::string id = "localhost:" + std::to_string(45000 + 37 * c0 % 9000);
    setenv("NEURON_RT_ROOT_COMM_ID", id.c_str(), 0);
  }
  bool loaded = false;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    int64_t id = -1; std::string reply;
    try {
      J req = parse_json(line.data(), line.size());
      if (auto x = req.get("id")) id = x->i;
      std::string op = req.get("op") ? req.get("op")->s : "";
      if (op == "hello") {
        if (!loaded) { setup(mdir); warmup(); loaded = true; }
        reply = "\"protocol\":1";
      } else if (op == "generate") {
        if (!loaded) { setup(mdir); warmup(); loaded = true; }
        std::vector<int64_t> prompt; if (auto p = req.get("prompt")) for (auto& v : p->a) prompt.push_back(v.i);
        int64_t mt = req.get("max_tokens") ? req.get("max_tokens")->i : 1;
        bool reset = req.get("reset") ? req.get("reset")->b : true;
        auto t0 = clk::now();
        auto toks = generate(prompt, mt, reset);
        fprintf(stderr, "[engine] generate plen=%zu max=%ld reset=%d: %.1f ms\n", prompt.size(), (long)mt, (int)reset, ms_since(t0));
        reply = "\"tokens\":["; for (size_t i = 0; i < toks.size(); i++) { if (i) reply += ","; reply += std::to_string(toks[i]); } reply += "]";
      } else if (op == "read") {  // development only: read bytes of a global tensor of a rank (hex-free: as float or int list)
        std::string name = req.get("name")->s; int rr = req.get("rank") ? (int)req.get("rank")->i : 0;
        int64_t off = req.get("offset")->i, n = req.get("n")->i; std::string ty = req.get("type")->s;
        nrt_tensor_t* t = name == "tokout" ? R[rr].tokout : name == "h0" ? R[rr].h[0] : name == "h1" ? R[rr].h[1] : name == "pos0" ? R[rr].pos[0] : name == "pos1" ? R[rr].pos[1] :
            name[0] == 'L' && name.find('_') != std::string::npos ? R[rr].L[std::stoi(name.substr(1))].t.at(name.substr(name.find('_') + 1)) : R[rr].g.at(name);
        int es = ty == "bf16" ? 2 : 4; std::vector<uint8_t> buf(n * es);
        CK(nrt_tensor_read(t, buf.data(), off * es, n * es));
        reply = "\"values\":[";
        for (int64_t i = 0; i < n; i++) {
          if (i) reply += ",";
          auto fs = [](float f) { return std::isnan(f) ? std::string("NaN") : std::isinf(f) ? std::string(f > 0 ? "Infinity" : "-Infinity") : std::to_string(f); };
          if (ty == "bf16") { uint16_t v; memcpy(&v, &buf[i * 2], 2); reply += fs(bf2f(v)); }
          else if (ty == "f32") { float v; memcpy(&v, &buf[i * 4], 4); reply += fs(v); }
          else { int32_t v; memcpy(&v, &buf[i * 4], 4); reply += std::to_string(v); }
        }
        reply += "]";
      } else if (op == "bench") {  // development only: time n queued executions of one NEFF
        if (!loaded) { setup(mdir); warmup(); loaded = true; }
        std::string what = req.get("what")->s; int n = (int)req.get("n")->i;
        auto t0 = clk::now();
        std::vector<std::string> errs(W); std::vector<std::thread> th;
        for (int r = 0; r < W; r++) th.emplace_back([&, r] {
          try {
            Rank& rk = R[r]; std::vector<NRT_STATUS> rets(n); nrta_seq_t seq = 0;
            for (int i = 0; i < n; i++) {
              Exec x;
              int lb = req.get("layer") ? (int)req.get("layer")->i : 0, cb = req.get("chunk") ? (int)req.get("chunk")->i : 0;
              if (what == "head") x = {rk.head.m, rk.head_in[0], rk.head_out[0]};
              else if (what == "pembed") x = {rk.pembed.m, rk.pembed_in[0], rk.pembed_out};
              else if (what == "pgather") x = {rk.pgather.m, rk.pgather_in[0], rk.pgather_out};
              else if (what.rfind("pmoe", 0) == 0) x = {rk.pmoe[std::stoi(what.substr(4))].m, rk.pmoe_in[lb], rk.pmoe_out};
              else if (what.rfind("pattn_s", 0) == 0) x = {rk.pattn_s[std::stoi(what.substr(7))].m, rk.pattn_in[lb], rk.pattn_out};
              else if (what.rfind("pattn_f", 0) == 0) x = {rk.pattn_f[std::stoi(what.substr(7))].m, rk.pattn_in[lb], rk.pattn_out};
              else x = {rk.step[std::stoi(what.substr(4))].m, rk.step_in[0], rk.step_out[0]};
              NRT_STATUS st;
              while ((st = nrta_execute_schedule(x.m, x.in, x.out, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {}
              CK(st);
            }
            bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
          } catch (std::exception& ex) { errs[r] = ex.what(); }
        });
        for (auto& t : th) t.join();
        for (auto& e : errs) if (!e.empty()) throw std::runtime_error(e);
        double ms = ms_since(t0);
        reply = "\"us_per_exec\":" + std::to_string(ms * 1000.0 / n);
      } else throw std::runtime_error("unknown op " + op);
      fprintf(g_out, "{\"id\":%ld,\"ok\":true,%s}\n", (long)id, reply.c_str());
    } catch (std::exception& ex) {
      fprintf(stderr, "[engine] error: %s\n", ex.what());
      fprintf(g_out, "{\"id\":%ld,\"ok\":false,\"error\":\"%s\"}\n", (long)id, jesc(ex.what()).c_str());
    }
    fflush(g_out);
  }
  if (loaded) nrt_close();  // also flushes runtime-captured profiles (NEURON_RT_INSPECT_*)
  return 0;
}
