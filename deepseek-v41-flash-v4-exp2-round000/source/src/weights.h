// Checkpoint access (mmap of the safetensors shards, values never changed) and the load-time re-layouts.
//
// Re-layouts (every value preserved exactly):
//  * dense fp8 e4m3fn + E8M0 32x32 block scales -> bf16 (fp8 * 2^k is exactly a bf16 value; the model card allows
//    this decode at load), tiled for the GEMV kernels
//  * routed-expert fp4 e2m1 (+ E8M0 per 32) -> legacy fp8 e4m3 bytes holding value * 2^-6 (exact for all 16 codes),
//    scales -> bf16 2^(e-127) * 2^6 (exact)
//  * bf16 / fp32 tensors -> tiled or row layouts
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "util.h"

struct TensorRef {
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t dim(int i) const { return shape.at(i); }
};

struct Checkpoint {
  std::map<std::string, TensorRef> t;
  std::vector<std::pair<void*, size_t>> maps;

  void open(const std::string& dir) {
    for (int i = 1; i <= 48; i++) {
      char nm[128];
      snprintf(nm, sizeof nm, "%s/model-%05d-of-00048.safetensors", dir.c_str(), i);
      int fd = ::open(nm, O_RDONLY);
      if (fd < 0) die(std::string("cannot open ") + nm);
      struct stat st;
      fstat(fd, &st);
      void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
      if (m == MAP_FAILED) die("mmap failed");
      ::close(fd);
      maps.push_back({m, (size_t)st.st_size});
      const uint8_t* base = (const uint8_t*)m;
      uint64_t hl;
      memcpy(&hl, base, 8);
      std::string hs((const char*)base + 8, hl);
      JVal h = json_parse(hs);
      for (auto& kv : h.o) {
        if (kv.first == "__metadata__") continue;
        TensorRef r;
        r.dtype = kv.second.get("dtype")->s;
        for (auto& d : kv.second.get("shape")->a) r.shape.push_back(d.inum);
        auto& off = kv.second.get("data_offsets")->a;
        r.data = base + 8 + hl + off[0].inum;
        r.nbytes = off[1].inum - off[0].inum;
        t[kv.first] = r;
      }
    }
  }
  const TensorRef& get(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) die("missing tensor " + n);
    return it->second;
  }
  bool has(const std::string& n) const { return t.count(n) > 0; }
};

// e4m3fn byte -> float (exact)
struct Fp8Lut {
  float v[256];
  Fp8Lut() {
    for (int b = 0; b < 256; b++) {
      int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
      float x = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8 + m), e - 10);
      if ((b & 0x7F) == 0x7F) x = NAN;
      v[b] = s ? -x : x;
    }
  }
};
static const Fp8Lut FP8LUT;

inline float e8m0(uint8_t e) { return ldexpf(1.0f, (int)e - 127); }

// Dense fp8 weight [N, K] (+ scale [N/32, K/32]) -> bf16 values W[n][k] (row-major, exact).
inline std::vector<uint16_t> dequant_fp8(const TensorRef& w, const TensorRef& s) {
  int64_t N = w.dim(0), K = w.dim(1);
  int64_t SK = s.dim(1);
  std::vector<uint16_t> out((size_t)N * K);
  for (int64_t n = 0; n < N; n++) {
    const uint8_t* row = w.data + n * K;
    const uint8_t* srow = s.data + (n / 32) * SK;
    uint16_t* o = &out[(size_t)n * K];
    for (int64_t k = 0; k < K; k++) o[k] = f2bf_exact(FP8LUT.v[row[k]] * e8m0(srow[k / 32]));
  }
  return out;
}

inline std::vector<uint16_t> as_bf16(const TensorRef& w) {
  if (w.dtype != "BF16") die("expected bf16");
  std::vector<uint16_t> v(w.nbytes / 2);
  memcpy(v.data(), w.data, w.nbytes);
  return v;
}

// tile_w: W [N, K] (elements of size E) -> [128, NT*KT*128]: out[p][(nt*KT + kt)*128 + j] = W[nt*128 + j][kt*128 + p]
template <typename T>
inline void tile_w(const T* W, int64_t N, int64_t K, T* out) {
  int64_t NT = N / 128, KT = K / 128;
  int64_t row = NT * KT * 128;
  for (int64_t nt = 0; nt < NT; nt++)
    for (int64_t kt = 0; kt < KT; kt++)
      for (int64_t j = 0; j < 128; j++) {
        const T* src = W + (nt * 128 + j) * K + kt * 128;
        T* dst = out + (nt * KT + kt) * 128 + j;
        for (int p = 0; p < 128; p++) dst[p * row] = src[p];
      }
}
template <typename T>
inline std::vector<T> tiled(const std::vector<T>& W, int64_t N, int64_t K) {
  std::vector<T> o(W.size());
  tile_w(W.data(), N, K, o.data());
  return o;
}

// x-stationary layout: W [N, K] -> [128, KT*N]: out[p][kt*N + n] = W[n][kt*128 + p]
inline std::vector<uint16_t> xstat(const std::vector<uint16_t>& W, int64_t N, int64_t K) {
  int64_t KT = K / 128;
  std::vector<uint16_t> o(W.size());
  for (int64_t p = 0; p < 128; p++)
    for (int64_t kt = 0; kt < KT; kt++)
      for (int64_t n = 0; n < N; n++) o[p * KT * N + kt * N + n] = W[n * K + kt * 128 + p];
  return o;
}

// hc fn [24, 20480] fp32 -> [128, 160*24]: out[p][kt*24 + n] = fn[n][kt*128 + p]
inline std::vector<float> tile_fn(const TensorRef& t) {
  int64_t N = t.dim(0), K = t.dim(1), KT = K / 128;
  const float* f = (const float*)t.data;
  std::vector<float> o((size_t)N * K);
  for (int64_t p = 0; p < 128; p++)
    for (int64_t kt = 0; kt < KT; kt++)
      for (int64_t n = 0; n < N; n++) o[p * KT * N + kt * N + n] = f[n * K + kt * 128 + p];
  return o;
}

// fp4 e2m1 nibble -> legacy fp8 e4m3 byte of value * 2^-6
static const uint8_t FP4TO8[16] = {0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C,
                                   0x80, 0x84, 0x88, 0x8C, 0x90, 0x94, 0x98, 0x9C};

// expert matrix: packed fp4 [N, K/2] -> tiled legacy fp8 [128, NT*KT*128] (written into out)
inline void expert_fp8_tiled(const TensorRef& w, uint8_t* out, std::vector<uint8_t>& scratch) {
  int64_t N = w.dim(0), K = w.dim(1) * 2;
  scratch.resize((size_t)N * K);
  for (int64_t i = 0; i < N * K / 2; i++) {
    uint8_t b = w.data[i];
    scratch[2 * i] = FP4TO8[b & 15];
    scratch[2 * i + 1] = FP4TO8[b >> 4];
  }
  tile_w(scratch.data(), N, K, out);
}

// expert scales E8M0 [N, K/32] -> bf16 2^(e-127+6)
// E8M0 scale bytes [N, NB] -> bf16 2^(e-127+6), laid out per partition: out[p][nt][b] = scale of row nt*128 + p
inline void expert_scales(const TensorRef& s, uint16_t* out) {
  const size_t N = (size_t)s.dim(0), NB = (size_t)s.dim(1), NT = N / 128;
  for (size_t n = 0; n < N; n++) {
    size_t p = n % 128, nt = n / 128;
    for (size_t b = 0; b < NB; b++) out[(p * NT + nt) * NB + b] = (uint16_t)(((uint16_t)s.data[n * NB + b] + 6) << 7);
  }
}
