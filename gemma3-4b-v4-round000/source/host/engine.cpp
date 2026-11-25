// Gemma-3-4B (text) inference engine for one logical NeuronCore (Trainium2, LNC=2).
// Host side: loads the checkpoint, re-lays out the weights, uploads them once, loads the embedded NEFFs and
// drives them through libnrt's async queue. The model itself runs in the NEFFs (kernels/gemma.py).
//
// Protocol: JSON lines on stdin/stdout (hello, generate). Diagnostics on stderr.
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "json.h"

// ------------------------------------------------------------------------------------------------ blobs
struct Blob { const char* name; const unsigned char* start; const unsigned char* end; };
#include "blobs.inc"   // generated: extern symbols and `static const Blob kBlobs[]`

static const Blob& blob(const std::string& name) {
  for (const auto& b : kBlobs) if (name == b.name) return b;
  throw std::runtime_error("missing embedded blob " + name);
}

// ------------------------------------------------------------------------------------------------ model constants
static constexpr int D = 2560, NL = 34, HD = 256, S = 8192, NTOK = S + 512, VOCAB = 262208;
static constexpr int TPRE = 256, TP2 = 512, VPC = 131200, VT = 1025;   // prefill chunks: 256 (one chunk), else 512
static const int DEC_GLOBAL_B[] = {512, 1024, 1536, 2048, 3072, 4096, 4608, 6144, 8192};
static const int PRE_LOCAL_B[] = {256, 512, 1024, 1280};
static const int PRE_GLOBAL_B[] = {256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 6144, 8192};
static constexpr int NPGB = sizeof(PRE_GLOBAL_B) / sizeof(int);
// prefill group NEFFs: (local bucket, global bucket) for chunk c, from need = (c + 1) * TPRE
static const int PRE_COMBOS[][2] = {{256, 256}, {512, 512}, {1024, 1024}, {1280, 2048}, {1280, 3072}, {1280, 4096},
                                    {1280, 6144}, {1280, 8192}};
// two-stream prefill NEFFs for 512-token chunks: need = (c + 1) * 512
static const int P2_COMBOS[][2] = {{512, 512}, {1024, 1024}, {1280, 1536}, {1280, 2048}, {1280, 2560}, {1280, 3072},
                                   {1280, 3584}, {1280, 4096}, {1280, 6144}, {1280, 8192}};
static const int DEC_STEPS = getenv("ENG_DEC1") ? 1 : getenv("ENG_DEC2") ? 2 : 4;   // max decode steps per NEFF execution
static constexpr int DEC4_MAXB = 8192;   // 4-step NEFFs for every global bucket
static constexpr int DEC2_MAXB = 8192;   // 2-step NEFFs for every global bucket
// the single 256-token chunk (single stream: two 128-token streams measured 18% slower)
static const char* PRE256 = "pgA256_256K";
// the last chunk's NEFF also runs the prompt's last token through layer 33 + LM head (else a separate hl33 NEFF)
static const bool FOLD = true;
static bool is_global(int l) { return l % 6 == 5; }

// ------------------------------------------------------------------------------------------------ utilities
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  char b_[512]; snprintf(b_, sizeof b_, "%s:%d %s -> %d (%s)", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); \
  throw std::runtime_error(b_);} } while (0)

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

static void parallel_for(size_t n, const std::function<void(size_t)>& f, int threads = 0) {
  if (threads <= 0) threads = std::min<int>(48, std::max(1u, std::thread::hardware_concurrency()));
  std::atomic<size_t> next(0);
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; t++) ts.emplace_back([&] { for (size_t i; (i = next++) < n;) f(i); });
  for (auto& t : ts) t.join();
}

// ------------------------------------------------------------------------------------------------ safetensors
struct TensorRef { const uint16_t* data; std::vector<int64_t> shape; };

struct SafeTensors {
  std::map<std::string, TensorRef> t;
  std::vector<std::pair<void*, size_t>> maps;
  void open_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st; fstat(fd, &st);
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("mmap failed " + path);
    maps.push_back({m, (size_t)st.st_size});
    const unsigned char* base = (const unsigned char*)m;
    uint64_t hlen; memcpy(&hlen, base, 8);
    json::Value hdr = json::parse(std::string((const char*)base + 8, hlen));
    for (auto& kv : hdr.obj) {
      if (kv.first == "__metadata__") continue;
      const json::Value& v = kv.second;
      if (v.at("dtype").str != "BF16") continue;
      TensorRef r;
      for (auto& d : v.at("shape").arr) r.shape.push_back((int64_t)d.num);
      r.data = (const uint16_t*)(base + 8 + hlen + (size_t)v.at("data_offsets").arr[0].num);
      t[kv.first] = r;
    }
  }
  const TensorRef& get(const std::string& name) const {
    auto it = t.find(name);
    if (it == t.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
  }
};

// Tile re-layout: rows[n] points at row n of the source (nullptr: zero row), coloff[kt] the column of k-tile kt.
// out[p][(nt*KTn + kt)*128 + j] = rows[nt*128 + j][coloff[kt] + p]
static void tiles(const std::vector<const uint16_t*>& rows, const std::vector<size_t>& coloff, uint16_t* out) {
  const size_t N = rows.size(), KTn = coloff.size(), NT = N / 128, W = NT * KTn * 128;
  parallel_for(NT, [&](size_t nt) {
    for (size_t kt = 0; kt < KTn; kt++) {
      const size_t t = nt * KTn + kt;
      for (size_t j = 0; j < 128; j++) {
        const uint16_t* src = rows[nt * 128 + j];
        if (src) { src += coloff[kt]; for (size_t p = 0; p < 128; p++) out[p * W + t * 128 + j] = src[p]; }
        else for (size_t p = 0; p < 128; p++) out[p * W + t * 128 + j] = 0;
      }
    }
  }, 16);
}

static std::vector<size_t> seqoff(int n) { std::vector<size_t> v(n); for (int i = 0; i < n; i++) v[i] = (size_t)i * 128; return v; }

// ------------------------------------------------------------------------------------------------ device
struct Neff {
  nrt_model_t* m = nullptr;
  std::vector<std::string> inputs, outputs;
  std::map<std::string, size_t> sizes;
};

static int g_vnc = 0;

static nrt_tensor_t* dev(size_t bytes, const char* name) {
  nrt_tensor_t* t;
  CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, g_vnc, bytes, name, &t));
  return t;
}

static Neff load_neff(const std::string& name) {
  const Blob& b = blob(name);
  Neff n;
  CK(nrt_load(b.start, (size_t)(b.end - b.start), g_vnc, 1, &n.m));
  nrt_tensor_info_array_t* info;
  CK(nrt_get_model_tensor_info(n.m, &info));
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& t = info->tensor_array[i];
    (t.usage == NRT_TENSOR_USAGE_INPUT ? n.inputs : n.outputs).push_back(t.name);
    n.sizes[t.name] = t.size;
  }
  nrt_free_model_tensor_info(info);
  return n;
}

struct Binding { std::map<std::string, nrt_tensor_t*> in, out; };

struct Sets { nrt_tensor_set_t* in = nullptr; nrt_tensor_set_t* out = nullptr; };

static std::map<nrt_tensor_t*, size_t> g_size;

static nrt_tensor_t* devz(size_t bytes, const char* name) {
  nrt_tensor_t* t = dev(bytes, name);
  CK(nrt_tensor_memset(t, 0, 0, bytes));
  g_size[t] = bytes;
  return t;
}

// Build tensor sets for a NEFF, checking every input/output of the NEFF is bound with the right size.
static Sets make_sets(const Neff& n, const Binding& b, const std::string& what) {
  Sets s;
  CK(nrt_allocate_tensor_set(&s.in));
  CK(nrt_allocate_tensor_set(&s.out));
  for (auto& name : n.inputs) {
    auto it = b.in.find(name);
    if (it == b.in.end()) throw std::runtime_error(what + ": unbound input " + name);
    if (g_size.count(it->second) && g_size[it->second] != n.sizes.at(name))
      throw std::runtime_error(what + ": size mismatch for " + name + " (" + std::to_string(g_size[it->second]) + " vs " +
                               std::to_string(n.sizes.at(name)) + ")");
    CK(nrt_add_tensor_to_tensor_set(s.in, name.c_str(), it->second));
  }
  for (auto& name : n.outputs) {
    auto it = b.out.find(name);
    if (it == b.out.end()) throw std::runtime_error(what + ": unbound output " + name);
    if (g_size.count(it->second) && g_size[it->second] != n.sizes.at(name))
      throw std::runtime_error(what + ": size mismatch for output " + name);
    CK(nrt_add_tensor_to_tensor_set(s.out, name.c_str(), it->second));
  }
  return s;
}

struct Engine {
  std::map<std::string, Neff> neffs;
  // weights
  nrt_tensor_t *emb, *embT, *lmask, *lids, *lnf, *cosl, *sinl, *cosg, *sing;
  struct LayerW { nrt_tensor_t *wqkv, *wo, *wgu, *wd, *lnw, *qkw, *kc, *vc; } L[NL];
  // state
  nrt_tensor_t *io, *toks, *pos[2], *zero_i, *pcol, *ppos, *dummy_i, *hdec[2], *hpre[2], *hp2[2], *out_dummy;
  std::vector<nrt_tensor_t*> spos, spos2;
  // sets
  Sets dec_all[2], last_pre, last_p2;
  std::vector<Sets> p2_all;   // [512-chunk]: all 34 layers in one NEFF
  Sets pre_all;               // the single 256-token chunk
  Sets pre_last;              // ... with the last token's layer 33 + LM head (FOLD)
  std::vector<Sets> p2_last;  // [512-chunk]: last chunk of a prompt + the last token's layer 33 + LM head (FOLD)
  // host state
  int len = 0;          // tokens consumed
  bool have_state = false;
  std::vector<int32_t> hostbuf;

  void load(const std::string& model_dir);
  std::vector<int32_t> generate(const std::vector<int64_t>& prompt, int n, bool reset);
};

static void upload(nrt_tensor_t* t, const void* p, size_t bytes) { CK(nrt_tensor_write(t, p, 0, bytes)); }

void Engine::load(const std::string& md) {
  auto t0 = clk::now();
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  uint32_t nv = 0; CK(nrt_get_visible_vnc_count(&nv));
  fprintf(stderr, "[engine] visible logical cores: %u\n", nv);

  // NEFFs
  // one 34-layer NEFF per prefill chunk (embedding, layers 0-32, layer 33's K/V), the last token's layer 33 + LM
  // head (hl33), one NEFF per one or two decode steps
  std::vector<std::string> names;
  for (auto& c : P2_COMBOS) {
    names.push_back("p2s512A" + std::to_string(c[0]) + "_" + std::to_string(c[1]) + "K");
    if (FOLD) names.push_back("p2s512A" + std::to_string(c[0]) + "_" + std::to_string(c[1]) + "KH");
  }
  for (int b : DEC_GLOBAL_B) { names.push_back("dgA" + std::to_string(b)); if (b <= DEC2_MAXB) names.push_back("dgA2x" + std::to_string(b)); if (b <= DEC4_MAXB) names.push_back("dgA4x" + std::to_string(b)); }
  names.push_back(PRE256);
  if (FOLD) names.push_back(std::string(PRE256) + "H");
  names.push_back("hl33_256");
  names.push_back("hl33_512");
  for (auto& n : names) neffs[n] = load_neff(n);
  fprintf(stderr, "[engine] %zu NEFFs loaded (%.0f ms)\n", neffs.size(), ms_since(t0));

  // checkpoint
  SafeTensors st;
  st.open_file(md + "/model-00001-of-00002.safetensors");
  st.open_file(md + "/model-00002-of-00002.safetensors");
  const std::string P = "language_model.model.";
  auto W = [&](const std::string& n) -> const TensorRef& { return st.get(P + n); };

  // embeddings (natural layout for gathers) and the LM head (tied, tiles per core)
  const TensorRef& E = W("embed_tokens.weight");
  if (E.shape[0] != VOCAB || E.shape[1] != D) throw std::runtime_error("unexpected embedding shape");
  emb = devz((size_t)VOCAB * D * 2, "emb");
  upload(emb, E.data, (size_t)VOCAB * D * 2);
  {
    const size_t per = (size_t)128 * VT * 20 * 128;
    std::vector<uint16_t> buf(2 * per);
    for (int c = 0; c < 2; c++) {
      std::vector<const uint16_t*> rows(VPC);
      for (int i = 0; i < VPC; i++) { size_t r = (size_t)VPC * c + i; rows[i] = r < (size_t)VOCAB ? E.data + r * D : nullptr; }
      tiles(rows, seqoff(20), buf.data() + c * per);
    }
    embT = devz(buf.size() * 2, "embT");
    upload(embT, buf.data(), buf.size() * 2);
    std::vector<float> mk(2 * 128 * VT), id(2 * 128 * VT);
    for (int c = 0; c < 2; c++) for (int p = 0; p < 128; p++) for (int t = 0; t < VT; t++) {
      long g = (long)VPC * c + (long)t * 128 + p;
      mk[(c * 128 + p) * VT + t] = g < VOCAB ? 0.f : -1e30f;
      id[(c * 128 + p) * VT + t] = (float)(1048576L - g);
    }
    lmask = devz(mk.size() * 4, "lmask"); upload(lmask, mk.data(), mk.size() * 4);
    lids = devz(id.size() * 4, "lids"); upload(lids, id.data(), id.size() * 4);
  }
  {  // norm weights partition-major: [p][kt] = w[kt * 128 + p] (one contiguous run per partition)
    std::vector<uint16_t> t(D);
    const uint16_t* w = W("norm.weight").data;
    for (int p = 0; p < 128; p++) for (int kt = 0; kt < 20; kt++) t[p * 20 + kt] = w[kt * 128 + p];
    lnf = devz(D * 2, "lnf"); upload(lnf, t.data(), D * 2);
  }
  {
    const Blob& tb = blob("rope_tables");
    const size_t one = (size_t)128 * S * 2;
    if ((size_t)(tb.end - tb.start) != 4 * one) throw std::runtime_error("bad rope table blob");
    cosl = devz(one, "cosl"); upload(cosl, tb.start, one);
    sinl = devz(one, "sinl"); upload(sinl, tb.start + one, one);
    cosg = devz(one, "cosg"); upload(cosg, tb.start + 2 * one, one);
    sing = devz(one, "sing"); upload(sing, tb.start + 3 * one, one);
  }
  fprintf(stderr, "[engine] embeddings ready (%.0f ms)\n", ms_since(t0));

  // layers
  const size_t n_qkv = (size_t)128 * 16 * 20 * 128, n_o = (size_t)128 * 20 * 8 * 128, n_gu = (size_t)128 * 80 * 20 * 128,
               n_d = (size_t)128 * 20 * 40 * 128;
  std::vector<uint16_t> bq(2 * n_qkv), bo(2 * n_o), bgu(2 * n_gu), bd(2 * n_d);
  for (int l = 0; l < NL; l++) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const TensorRef &q = W(p + "self_attn.q_proj.weight"), &k = W(p + "self_attn.k_proj.weight"),
                    &v = W(p + "self_attn.v_proj.weight"), &o = W(p + "self_attn.o_proj.weight"),
                    &g = W(p + "mlp.gate_proj.weight"), &u = W(p + "mlp.up_proj.weight"), &dn = W(p + "mlp.down_proj.weight");
    for (int c = 0; c < 2; c++) {
      std::vector<const uint16_t*> rows;
      for (int i = 0; i < 1024; i++) rows.push_back(q.data + (size_t)(1024 * c + i) * D);
      for (int i = 0; i < 512; i++) rows.push_back(k.data + (size_t)(512 * c + i) * D);
      for (int i = 0; i < 512; i++) rows.push_back(v.data + (size_t)(512 * c + i) * D);
      tiles(rows, seqoff(20), bq.data() + c * n_qkv);
      rows.clear();
      for (int i = 0; i < D; i++) rows.push_back(o.data + (size_t)i * 2048);
      std::vector<size_t> co(8);
      for (int kt = 0; kt < 8; kt++) co[kt] = (size_t)(4 * c + kt % 4) * 256 + (kt / 4) * 128;
      tiles(rows, co, bo.data() + c * n_o);
      rows.clear();
      for (int nt = 0; nt < 40; nt++) {
        for (int i = 0; i < 128; i++) rows.push_back(g.data + (size_t)(5120 * c + nt * 128 + i) * D);
        for (int i = 0; i < 128; i++) rows.push_back(u.data + (size_t)(5120 * c + nt * 128 + i) * D);
      }
      tiles(rows, seqoff(20), bgu.data() + c * n_gu);
      rows.clear();
      for (int i = 0; i < D; i++) rows.push_back(dn.data + (size_t)i * 10240);
      std::vector<size_t> cd(40);
      for (int kt = 0; kt < 40; kt++) cd[kt] = (size_t)5120 * c + kt * 128;
      tiles(rows, cd, bd.data() + c * n_d);
    }
    LayerW& lw = L[l];
    lw.wqkv = devz(bq.size() * 2, "wqkv"); upload(lw.wqkv, bq.data(), bq.size() * 2);
    lw.wo = devz(bo.size() * 2, "wo"); upload(lw.wo, bo.data(), bo.size() * 2);
    lw.wgu = devz(bgu.size() * 2, "wgu"); upload(lw.wgu, bgu.data(), bgu.size() * 2);
    lw.wd = devz(bd.size() * 2, "wd"); upload(lw.wd, bd.data(), bd.size() * 2);
    std::vector<uint16_t> ln(4 * D);
    const char* lnn[4] = {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"};
    for (int i = 0; i < 4; i++) {   // [p][i][kt] = w_i[kt * 128 + p]
      const uint16_t* w = W(p + lnn[i] + ".weight").data;
      for (int q = 0; q < 128; q++) for (int kt = 0; kt < 20; kt++) ln[(q * 4 + i) * 20 + kt] = w[kt * 128 + q];
    }
    lw.lnw = devz(ln.size() * 2, "lnw"); upload(lw.lnw, ln.data(), ln.size() * 2);
    std::vector<uint16_t> qk(2 * HD);
    {  // [p][n][dt] = w_n[dt * 128 + p]
      const uint16_t* wq = W(p + "self_attn.q_norm.weight").data;
      const uint16_t* wk = W(p + "self_attn.k_norm.weight").data;
      for (int q = 0; q < 128; q++) for (int dt = 0; dt < 2; dt++) {
        qk[q * 4 + dt] = wq[dt * 128 + q];
        qk[q * 4 + 2 + dt] = wk[dt * 128 + q];
      }
    }
    lw.qkw = devz(qk.size() * 2, "qkw"); upload(lw.qkw, qk.data(), qk.size() * 2);
    lw.kc = devz((size_t)4 * HD * S * 2, "kc");
    lw.vc = devz((size_t)S * 1024 * 2, "vc");
  }
  fprintf(stderr, "[engine] layers ready (%.0f ms)\n", ms_since(t0));

  // state tensors
  // toks, pcol and ppos are slices of one tensor so that a prefill request needs a single host write
  io = devz((size_t)(NTOK + 4) * 4, "io");
  CK(nrt_tensor_allocate_slice(io, 0, 4, "pcol", &pcol)); g_size[pcol] = 4;
  CK(nrt_tensor_allocate_slice(io, 4, 4, "ppos", &ppos)); g_size[ppos] = 4;
  CK(nrt_tensor_allocate_slice(io, 16, (size_t)NTOK * 4, "toks", &toks)); g_size[toks] = (size_t)NTOK * 4;
  for (int i = 0; i < 2; i++) { pos[i] = devz(4, "pos"); hdec[i] = devz((size_t)D * 2, "hdec"); hpre[i] = devz((size_t)D * TPRE * 2, "hpre");
                               hp2[i] = devz((size_t)D * TP2 * 2, "hp2"); }
  zero_i = devz(4, "zero"); dummy_i = devz(4, "dummy");
  out_dummy = devz(8, "outd");
  for (int c = 0; c < S / TPRE; c++) {
    nrt_tensor_t* t = devz(4, "spos");
    int32_t v = c * TPRE; upload(t, &v, 4);
    spos.push_back(t);
  }
  for (int c = 0; c < S / TP2; c++) {
    nrt_tensor_t* t = devz(4, "spos2");
    int32_t v = c * TP2; upload(t, &v, 4);
    spos2.push_back(t);
  }

  // tensor sets
  // decode step: group F = embedding + layers 0-11, M = layers 12-23, T = layers 24-33 + LM head
  auto group_bind = [&](Binding& b, int l0, int n) {
    static const char* lwn[8] = {"wqkv", "wo", "wgu", "wd", "lnw", "qkw", "kc", "vc"};
    for (int i = 0; i < n; i++) {
      const LayerW& lw = L[l0 + i];
      nrt_tensor_t* t[8] = {lw.wqkv, lw.wo, lw.wgu, lw.wd, lw.lnw, lw.qkw, lw.kc, lw.vc};
      for (int k = 0; k < 8; k++) b.in["l" + std::to_string(i) + "_" + lwn[k]] = t[k];
    }
    b.in["cosl"] = cosl; b.in["sinl"] = sinl; b.in["cosg"] = cosg; b.in["sing"] = sing;
  };
  for (int par = 0; par < 2; par++) {   // one NEFF per decode step: embedding, layers 0-33, LM head + argmax
    Binding b;
    group_bind(b, 0, NL);
    b.in["pos"] = pos[par]; b.in["toks"] = toks; b.in["emb"] = emb; b.in["pnext"] = pos[1 - par];
    b.in["lnf"] = lnf; b.in["embT"] = embT; b.in["lmask"] = lmask; b.in["lids"] = lids;
    b.out = {{"output_0", out_dummy}};
    dec_all[par] = make_sets(neffs["dgA512"], b, "dec step");
  }
  {
    Binding b;
    group_bind(b, 0, NL);
    b.in["pos"] = spos[0]; b.in["toks"] = toks; b.in["emb"] = emb;
    b.out = {{"output_0", hpre[1]}};
    pre_all = make_sets(neffs[PRE256], b, "pre all");
  }
  {  // the prompt's last token through layer 33 (the prefill groups' last layer only writes its K/V) + LM head
    Binding h;
    group_bind(h, NL - 1, 1);
    h.in["col"] = pcol; h.in["pos"] = ppos; h.in["pnext"] = dummy_i; h.in["toks"] = toks; h.in["lnf"] = lnf;
    h.in["embT"] = embT; h.in["lmask"] = lmask; h.in["lids"] = lids;
    h.out = {{"output_0", out_dummy}};
    h.in["h"] = hpre[1];
    last_pre = make_sets(neffs["hl33_256"], h, "hl33_256");
    h.in["h"] = hp2[1];
    last_p2 = make_sets(neffs["hl33_512"], h, "hl33_512");
  }
  for (int c = 0; c < S / TP2; c++) {
    Binding b;
    group_bind(b, 0, NL);
    b.in["pos"] = spos2[c]; b.in["toks"] = toks; b.in["emb"] = emb;
    b.out = {{"output_0", hp2[1]}};
    p2_all.push_back(make_sets(neffs["p2s512A512_512K"], b, "p2 all"));
  }
  if (FOLD) {   // last chunk + the last token's layer 33, LM head and argmax in one NEFF
    auto lastb = [&](Binding& b, nrt_tensor_t* p) {
      group_bind(b, 0, NL);
      b.in["pos"] = p; b.in["toks"] = toks; b.in["emb"] = emb; b.in["col"] = pcol; b.in["tpos"] = ppos;
      b.in["pnext"] = dummy_i; b.in["lnf"] = lnf; b.in["embT"] = embT; b.in["lmask"] = lmask; b.in["lids"] = lids;
      b.out = {{"output_0", out_dummy}};
    };
    Binding b;
    lastb(b, spos[0]);
    pre_last = make_sets(neffs[std::string(PRE256) + "H"], b, "pre last");
    for (int c = 0; c < S / TP2; c++) {
      Binding b2;
      lastb(b2, spos2[c]);
      p2_last.push_back(make_sets(neffs["p2s512A512_512KH"], b2, "p2 last"));
    }
  }
  hostbuf.resize(NTOK);
  fprintf(stderr, "[engine] ready (%.0f ms)\n", ms_since(t0));
}

// ------------------------------------------------------------------------------------------------ scheduling
struct Queue {
  std::vector<NRT_STATUS> rets;
  nrta_seq_t last = 0;
  size_t n = 0;
  explicit Queue(size_t cap) : rets(cap, NRT_SUCCESS) {}
  void run(const Neff& m, const Sets& s) {
    if (n >= rets.size()) throw std::runtime_error("queue capacity exceeded");
    NRT_STATUS st;
    while ((st = nrta_execute_schedule(m.m, s.in, s.out, 0, &rets[n], &last)) == NRT_QUEUE_FULL) {}
    CK(st);
    n++;
  }
  void wait() {
    if (!n) return;
    bool done = false;
    while (!done) CK(nrta_is_completed(last, &done));
    for (size_t i = 0; i < n; i++) if (rets[i] != NRT_SUCCESS) throw std::runtime_error("execution failed: " + std::to_string((int)rets[i]));
  }
};

static int pick(const int* b, int nb, int need) {
  for (int i = 0; i < nb; i++) if (b[i] >= need) return b[i];
  return -1;
}

std::vector<int32_t> Engine::generate(const std::vector<int64_t>& prompt, int n, bool reset) {
  if (n < 1) throw std::runtime_error("max_tokens must be at least 1");
  int start;   // position of the first decode step
  int ndec;    // decode steps
  Queue q((size_t)(n + 2) * (NL + 2) + (size_t)(S / TPRE) * (NL + 1) + 8);
  if (reset) {
    const int plen = (int)prompt.size();
    if (plen < 1) throw std::runtime_error("a reset request needs a prompt");
    if (plen + n - 1 > S) throw std::runtime_error("prompt + max_tokens exceeds the engine's context (" + std::to_string(S) + ")");
    for (auto t : prompt) if (t < 0 || t >= VOCAB) throw std::runtime_error("token id out of range");
    const int CH = plen <= TPRE ? TPRE : TP2;
    const int nch = (plen + CH - 1) / CH;
    std::vector<int32_t> tk(4 + (size_t)nch * CH, 0);   // [pcol, ppos, -, -, tokens ...] = the io tensor's prefix
    tk[0] = plen - 1 - (nch - 1) * CH;
    tk[1] = plen - 1;
    for (int i = 0; i < plen; i++) tk[4 + i] = (int32_t)prompt[i];
    CK(nrt_tensor_write(io, tk.data(), 0, tk.size() * 4));
    for (int c = 0; c < nch; c++) {
      const int need = (c + 1) * CH;
      const int bl = pick(PRE_LOCAL_B, 4, std::min(need, 1280)), bg = pick(PRE_GLOBAL_B, NPGB, need);
      const std::string k = std::to_string(bl) + "_" + std::to_string(bg);
      const bool fold = FOLD && c == nch - 1;
      if (CH == TPRE) q.run(neffs.at(std::string(PRE256) + (fold ? "H" : "")), fold ? pre_last : pre_all);
      else q.run(neffs.at("p2s512A" + k + (fold ? "KH" : "K")), fold ? p2_last[c] : p2_all[c]);
    }
    if (!FOLD) q.run(neffs.at(CH == TPRE ? "hl33_256" : "hl33_512"), CH == TPRE ? last_pre : last_p2);
    start = plen;
    ndec = n - 1;
    len = plen;
  } else {
    if (!have_state) throw std::runtime_error("a continuation needs an earlier generate request");
    if (!prompt.empty()) throw std::runtime_error("a continuation has an empty prompt");
    if (len + n > S) throw std::runtime_error("continuation exceeds the engine's context (" + std::to_string(S) + ")");
    start = len;
    ndec = n;
  }
  if (ndec > 0) {
    int32_t p0 = start;
    CK(nrt_tensor_write(pos[0], &p0, 0, 4));
    // dgA4x / dgA2x: four / two decode steps per NEFF; each execution reads pos[par] and writes pos[1 - par]
    for (int i = 0, par = 0; i < ndec; par ^= 1) {
      const int p = start + i;
      if (DEC_STEPS >= 4 && ndec - i >= 4 && p + 4 <= DEC4_MAXB) {
        q.run(neffs.at("dgA4x" + std::to_string(pick(DEC_GLOBAL_B, 9, p + 4))), dec_all[par]);
        i += 4;
      } else if (DEC_STEPS >= 2 && ndec - i >= 2 && p + 2 <= DEC2_MAXB) {
        q.run(neffs.at("dgA2x" + std::to_string(pick(DEC_GLOBAL_B, 9, p + 2))), dec_all[par]);
        i += 2;
      } else {
        q.run(neffs.at("dgA" + std::to_string(pick(DEC_GLOBAL_B, 9, p + 1))), dec_all[par]);
        i += 1;
      }
    }
  }
  q.wait();
  // tokens emitted: toks[first .. first + n), first = start + 1 - (reset ? 1 : 0)... (reset: toks[plen..], else toks[len+1..])
  const int first = reset ? start : start + 1;
  std::vector<int32_t> out(n);
  CK(nrt_tensor_read(toks, out.data(), (size_t)first * 4, (size_t)n * 4));
  len = reset ? start + n - 1 : len + n;
  have_state = true;
  for (auto t : out) if (t < 0 || t >= VOCAB) throw std::runtime_error("device produced an invalid token");
  return out;
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
  if (model.empty()) { fprintf(stderr, "usage: %s --model MODEL_DIR\n", argv[0]); return 2; }
  // replies go to a private copy of stdout; anything else printing to fd 1 lands on stderr
  int out_fd = dup(1);
  dup2(2, 1);
  FILE* out = fdopen(out_fd, "w");
  Engine eng;
  bool loaded = false;
  std::string line;
  char buf[1 << 16];
  std::string acc;
  auto reply = [&](const std::string& s) { fwrite(s.data(), 1, s.size(), out); fputc('\n', out); fflush(out); };
  for (;;) {
    // spin (non-blocking poll) for up to 10 s after the last input before blocking: a process woken from a
    // blocking read() answers the next request later
    {
      auto t_spin = clk::now();
      for (;;) {
        struct pollfd pfd = {0, POLLIN, 0};
        if (poll(&pfd, 1, 0) > 0) break;
        if (ms_since(t_spin) > 10000.0) break;
      }
    }
    ssize_t r = read(0, buf, sizeof buf);
    if (r <= 0) break;
    acc.append(buf, r);
    size_t nl;
    while ((nl = acc.find('\n')) != std::string::npos) {
      line = acc.substr(0, nl);
      acc.erase(0, nl + 1);
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      long long id = -1;
      bool has_id = false;
      try {
        json::Value req = json::parse(line);
        if (req.has("id")) { id = (long long)req.at("id").num; has_id = true; }
        const std::string op = req.at("op").str;
        if (op == "hello") {
          if (!loaded) { eng.load(model); loaded = true; }
          reply("{\"id\":" + std::to_string(id) + ",\"ok\":true,\"protocol\":1}");
        } else if (op == "generate") {
          if (!loaded) { eng.load(model); loaded = true; }
          std::vector<int64_t> prompt;
          if (req.has("prompt")) for (auto& v : req.at("prompt").arr) prompt.push_back((int64_t)v.num);
          int n = (int)req.at("max_tokens").num;
          bool reset = req.has("reset") ? req.at("reset").b : true;
          auto t0 = clk::now();
          auto toks = eng.generate(prompt, n, reset);
          fprintf(stderr, "[engine] generate plen=%zu n=%d reset=%d: %.2f ms\n", prompt.size(), n, (int)reset, ms_since(t0));
          std::string s = "{\"id\":" + std::to_string(id) + ",\"ok\":true,\"tokens\":[";
          for (size_t i = 0; i < toks.size(); i++) { if (i) s += ','; s += std::to_string(toks[i]); }
          s += "]}";
          reply(s);
        } else {
          throw std::runtime_error("unknown op " + op);
        }
      } catch (const std::exception& e) {
        fprintf(stderr, "[engine] error: %s\n", e.what());
        std::string msg = json::escape(e.what());
        reply("{\"id\":" + (has_id ? std::to_string(id) : std::string("null")) + ",\"ok\":false,\"error\":\"" + msg + "\"}");
      }
    }
  }
  return 0;
}
