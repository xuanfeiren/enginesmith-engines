// idle-penalty micro-benchmark: model A (big) on core c; vary the idle time before it and an optional warm-up
// launch of model W (tiny) or of A itself X us before.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { NRT_STATUS s_=(x); if (s_) { fprintf(stderr, "%s -> %d\n", #x, (int)s_); exit(1);} } while(0)
struct M { nrt_model_t* m; nrt_tensor_set_t *in, *out; };
static M load(const char* path, int core) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  M r; CK(nrt_load(b.data(), b.size(), core, 1, &r.m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(r.m, &info));
  CK(nrt_allocate_tensor_set(&r.in)); CK(nrt_allocate_tensor_set(&r.out));
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& t = info->tensor_array[i];
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, core, t.size, t.name, &x));
    CK(nrt_tensor_memset(x, 0, 0, t.size));
    if (t.usage == NRT_TENSOR_USAGE_INPUT) CK(nrt_add_tensor_to_tensor_set(r.in, t.name, x));
    else CK(nrt_add_tensor_to_tensor_set(r.out, t.name, x));
  }
  return r;
}
static NRT_STATUS g_ret[4096]; static int g_ri = 0;
static nrta_seq_t launch(M& m) { NRT_STATUS& ret = g_ret[(g_ri++) % 4096]; nrta_seq_t seq; while (true) { NRT_STATUS s = nrta_execute_schedule(m.m, m.in, m.out, 0, &ret, &seq); if (s == NRT_SUCCESS) break; if (s != NRT_QUEUE_FULL) { fprintf(stderr, "sched %d\n", s); exit(1);} } return seq; }
static void waitc(nrta_seq_t s) { bool d = false; while (!d) CK(nrta_is_completed(s, &d)); }
static void spin(double us) { double t = now() + us * 1e-6; while (now() < t) {} }

int main(int argc, char** argv) {
  int core = atoi(argv[1]);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  M A = load(argv[2], core); M B = load(argv[3], core);
  // second tensor set for A (same model, other buffers): does a tensor-set switch cost too?
  M A2 = load(argv[2], core);
  for (int i = 0; i < 30; i++) { waitc(launch(A)); waitc(launch(B)); }
  for (int mode = 0; mode < 4; mode++) {   // 0: A only, 1: A,B alternating, 2: A, A2(other load of same neff) alternating, 3: A then B queued right after A
    std::vector<double> tw, tsv;
    for (int it = 0; it < 80; it++) {
      spin(2000);
      double t0 = now();
      nrta_seq_t sq = launch(A);
      double t1 = now();
      waitc(sq);
      tw.push_back((now() - t0) * 1e6); tsv.push_back((t1 - t0) * 1e6);
      if (mode == 1) { spin(2000); waitc(launch(B)); }
      if (mode == 2) { spin(2000); waitc(launch(A2)); }
    }
    std::sort(tw.begin(), tw.end()); std::sort(tsv.begin(), tsv.end());
    printf("mode %d: A total p50 %7.1f p10 %7.1f p90 %7.1f  sched p50 %6.1f p90 %6.1f\n", mode, tw[40], tw[8], tw[72], tsv[40], tsv[72]);
    fflush(stdout);
  }
  nrt_close();
}
